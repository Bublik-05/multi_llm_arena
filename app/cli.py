"""Manual smoke-test CLI for the LLM provider.

python -m app.cli health
python -m app.cli models
python -m app.cli info llama3.2
python -m app.cli ask llama3.2 "Explain TCP in two sentences"
"""

import argparse
import asyncio
import sys
import time

from app.core.config import get_settings
from app.core.constants import CRITERIA_WEIGHTS, MIN_VALID_RESPONSES
from app.core.errors import ArenaError, InvalidJudgeResponseError
from app.core.logging import setup_logging
from app.services.evaluation.evaluation_service import EvaluationService
from app.services.evaluation.judge_service import JudgeService
from app.services.evaluation.scoring import rank_responses
from app.services.llm.factory import create_provider
from app.services.llm.types import GenerationRequest


def _out(text: str = "") -> None:
    sys.stdout.write(text + "\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app.cli", description=__doc__.split("\n")[0])
    parser.add_argument("-v", "--verbose", action="store_true", help="show INFO logs")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("health", help="check that Ollama is reachable")
    sub.add_parser("models", help="list installed models")
    info = sub.add_parser("info", help="show details of one model")
    info.add_argument("model")
    ask = sub.add_parser("ask", help="send a prompt to one model")
    ask.add_argument("model")
    ask.add_argument("prompt")
    ask.add_argument("--temperature", type=float, default=None)
    ask.add_argument("--max-tokens", type=int, default=None)
    ask.add_argument("--think", choices=["on", "off"], default=None, help="toggle thinking mode")
    collect = sub.add_parser("collect", help="send one prompt to several models in parallel")
    collect.add_argument("prompt")
    collect.add_argument("models", nargs="+")
    compare = sub.add_parser("compare", help="collect, judge and rank (nothing is saved)")
    compare.add_argument("prompt")
    compare.add_argument("models", nargs="+")
    compare.add_argument("--judge", default=None, help="default: DEFAULT_JUDGE_MODEL from .env")
    return parser


async def _collect(provider, args: argparse.Namespace) -> int:
    service = EvaluationService(provider, get_settings())
    start = time.perf_counter()
    runs = await service.collect_responses(args.prompt, args.models)
    wall = time.perf_counter() - start
    for run in runs:
        if run.ok:
            _out(f"[ok]    {run.model}  latency={run.latency_ms / 1000:.2f}s")
            _out("        " + (run.text or "")[:300].replace("\n", "\n        "))
        else:
            _out(f"[{run.status.value}] {run.model}  {run.error_code}: {run.error}")
    total = sum(r.latency_ms for r in runs) / 1000
    _out(f"\nwall time {wall:.2f}s (sum of individual latencies {total:.2f}s)")
    return 0 if any(r.ok for r in runs) else 1


async def _compare(provider, args: argparse.Namespace) -> int:
    settings = get_settings()
    judge_model = args.judge or settings.default_judge_model
    runs = await EvaluationService(provider, settings).collect_responses(args.prompt, args.models)
    for run in runs:
        if not run.ok:
            _out(f"[{run.status.value}] {run.model}  {run.error}")
    valid = [r for r in runs if r.ok]
    if len(valid) < MIN_VALID_RESPONSES:
        _out(f"Need at least {MIN_VALID_RESPONSES} valid responses to judge, got {len(valid)}.")
        return 1

    try:
        result = await JudgeService(provider, settings).evaluate(
            args.prompt, {r.model: r.text or "" for r in valid}, judge_model
        )
    except InvalidJudgeResponseError as exc:
        _out(f"Judge failed: {exc.message}")
        _out("Last raw output:\n" + str(exc.details.get("raw", ""))[:1000])
        return 1

    # Вывод сгенерированных ответов с правильными отступами
    _out("\n--- Сгенерированные ответы моделей ---")
    for label, model_name in result.label_map.items():
        resp_text = next((r.text for r in valid if r.model == model_name), "")
        _out(f"\n[Response {label}] ({model_name}):")
        _out(resp_text)

    ranked = rank_responses(result.scores, {r.model: r.latency_ms for r in valid})
    latency = {r.model: r.latency_ms for r in valid}
    short = {name: name[:4] for name in CRITERIA_WEIGHTS}
    header = "  ".join(f"{short[c]:>4}" for c in CRITERIA_WEIGHTS)
    _out(f"\njudge={judge_model}  attempts={result.attempts}")
    _out(f"{'#':<3}{'MODEL':<28}{header}  {'OVERALL':>7}  {'LATENCY':>7}")
    for item in ranked:
        cells = "  ".join(f"{getattr(item.scores, c):>4.0f}" for c in CRITERIA_WEIGHTS)
        _out(
            f"{item.rank:<3}{item.model:<28}{cells}  {item.overall:>7.2f}  "
            f"{latency[item.model] / 1000:>6.1f}s"
        )
    _out(f"\nlabels (as the judge saw them): {result.label_map}")
    _out(f"judge reasoning: {result.reasoning}")
    if result.truncated_models:
        _out(f"note: responses truncated to fit the judge's context: {result.truncated_models}")
    return 0


async def _run(args: argparse.Namespace) -> int:
    provider = create_provider(get_settings())
    try:
        if args.command == "health":
            health = await provider.health_check()
            _out(f"available={health.available} version={health.version} detail={health.detail}")
            return 0 if health.available else 1

        if args.command == "models":
            models = await provider.list_models()
            if not models:
                _out("No models installed. Try: ollama pull llama3.2")
            for m in models:
                size = f"{m.size_bytes / 1e9:.1f} GB" if m.size_bytes else "?"
                _out(f"{m.name:<32} {m.parameter_size or '?':<8} {m.quantization or '?':<8} {size}")
            return 0

        if args.command == "info":
            _out((await provider.get_model_info(args.model)).model_dump_json(indent=2))
            return 0

        if args.command == "collect":
            return await _collect(provider, args)

        if args.command == "compare":
            return await _compare(provider, args)

        think = None if args.think is None else args.think == "on"
        result = await provider.generate(
            GenerationRequest(
                model=args.model,
                prompt=args.prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                think=think,
            )
        )
        _out(result.text)
        _out()
        _out(
            f"-- latency={result.latency_ms / 1000:.2f}s "
            f"load={(result.load_ms or 0) / 1000:.2f}s "
            f"tokens={result.completion_tokens} "
            f"speed={result.tokens_per_sec or 0:.1f} tok/s"
        )
        return 0
    except ArenaError as exc:
        _out(f"Error [{exc.code}]: {exc.message}")
        return 1
    finally:
        await provider.aclose()


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    # Windows consoles may not support every character a model emits.
    sys.stdout.reconfigure(errors="replace")  # type: ignore[union-attr]
    setup_logging("INFO" if args.verbose else "WARNING")
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
