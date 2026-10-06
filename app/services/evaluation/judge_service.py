"""Judge: anonymises responses, asks a model to score them, validates the structured output."""

import json
import logging
import random
import re
import secrets
import string
from collections.abc import Mapping

from pydantic import BaseModel, ValidationError

from app.core.config import Settings
from app.core.constants import (
    CHARS_PER_TOKEN,
    CONTEXT_SAFETY_TOKENS,
    FEEDBACK_RESERVE_CHARS,
    JUDGE_MAX_ATTEMPTS,
    MAX_REASONING_CHARS,
    MIN_VALID_RESPONSES,
)
from app.core.errors import EmptyResponseError, InvalidJudgeResponseError
from app.schemas.judge import CriterionScores, JudgeOutput, build_judge_schema
from app.services.evaluation.prompts import (
    build_system_prompt,
    build_user_prompt,
    fit_texts,
)
from app.services.llm.base import LLMProvider
from app.services.llm.types import GenerationRequest
from app.services.llm.utils import strip_think_blocks

logger = logging.getLogger(__name__)


class JudgeResult(BaseModel):
    scores: dict[str, CriterionScores]  # keyed by real model name (de-anonymised)
    reasoning: str  # written by the judge; refers to responses by label
    label_map: dict[str, str]  # anonymous label -> model, needed to read `reasoning`
    raw: str  # judge's final raw output, kept for debugging/persistence
    attempts: int
    truncated_models: list[str]  # responses shortened to fit the judge's context window


# ---------------------------------------------------------------------- parsing
def extract_json(raw: str) -> object:
    """Parse JSON from judge output, tolerating code fences and surrounding prose."""
    text = strip_think_blocks(raw).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise InvalidJudgeResponseError("Output is not valid JSON")


def _normalize_label(key: str) -> str:
    """Accept 'a', 'Response A', 'Answer B:' as labels."""
    key = re.sub(r"^(RESPONSE|ANSWER)\s+", "", key.strip().upper())
    return key.strip(" :.")


def _summarize(exc: ValidationError) -> str:
    parts = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:3]]
    return "Schema validation failed: " + "; ".join(parts)


def parse_judge_output(raw: str, labels: list[str]) -> JudgeOutput:
    """Validate raw judge text. Raises InvalidJudgeResponseError with a retry-friendly message.

    Every expected label must be present with all criteria in range. Extra labels or fields
    are ignored.
    """
    data = extract_json(raw)
    if not isinstance(data, dict):
        raise InvalidJudgeResponseError("JSON root must be an object")
    try:
        output = JudgeOutput.model_validate(data)
    except ValidationError as exc:
        raise InvalidJudgeResponseError(_summarize(exc)) from exc

    normalized = {_normalize_label(k): v for k, v in output.scores.items()}
    missing = [label for label in labels if label not in normalized]
    if missing:
        raise InvalidJudgeResponseError(f"Missing scores for labels: {', '.join(missing)}")
    return JudgeOutput(
        reasoning=output.reasoning[:MAX_REASONING_CHARS],
        scores={label: normalized[label] for label in labels},
    )


# ---------------------------------------------------------------------- service
class JudgeService:
    def __init__(
        self,
        provider: LLMProvider,
        settings: Settings,
        *,
        rng: random.Random | None = None,
        max_attempts: int = JUDGE_MAX_ATTEMPTS,
    ) -> None:
        self._provider = provider
        self._settings = settings
        self._rng = rng or random.Random()  # injectable for deterministic tests
        self._max_attempts = max_attempts
        self._system_prompt = build_system_prompt()

    async def evaluate(
        self, prompt: str, responses: Mapping[str, str], judge_model: str
    ) -> JudgeResult:
        """Score `responses` (model name -> text) without revealing model names to the judge.

        Raises InvalidJudgeResponseError if no valid output after all attempts (details carry
        the raw output). Provider errors such as timeouts propagate unchanged and are not
        retried: a retry would only repeat a slow or failing call.
        """
        if len(responses) < MIN_VALID_RESPONSES:
            raise ValueError(f"Judging needs at least {MIN_VALID_RESPONSES} responses")
        if len(responses) > len(string.ascii_uppercase):
            raise ValueError("Too many responses to label with letters")

        # Shuffle first, then label by position: hides model identity and removes position bias.
        order = list(responses)
        self._rng.shuffle(order)
        label_map = dict(zip(string.ascii_uppercase, order, strict=False))
        labels = list(label_map)
        label_of = {model: label for label, model in label_map.items()}
        logger.info("Starting judge evaluation judge=%s responses=%d", judge_model, len(labels))

        nonce = secrets.token_hex(4)
        texts, truncated_labels = fit_texts(
            {label: responses[model] for label, model in label_map.items()},
            self._response_budget_chars(prompt, labels, nonce),
        )
        if truncated_labels:
            logger.warning(
                "Responses truncated to fit judge context labels=%s (num_ctx=%d)",
                truncated_labels,
                self._settings.judge_num_ctx,
            )
        user_prompt = build_user_prompt(prompt, texts, nonce)
        schema = build_judge_schema(labels)

        errors: list[str] = []
        raw = ""
        for attempt in range(1, self._max_attempts + 1):
            request = GenerationRequest(
                model=judge_model,
                system=self._system_prompt,
                prompt=user_prompt + self._feedback(errors),
                temperature=0,
                max_tokens=self._settings.judge_max_tokens,
                num_ctx=self._settings.judge_num_ctx,
                json_schema=schema,
                think=False,  # we want a short, deterministic JSON, not a reasoning trace
                timeout=self._settings.request_timeout,
            )
            try:
                raw = (await self._provider.generate(request)).text
                parsed = parse_judge_output(raw, labels)
            except (EmptyResponseError, InvalidJudgeResponseError) as exc:
                errors.append(exc.message)
                logger.warning(
                    "Judge response invalid attempt=%d/%d error=%s",
                    attempt,
                    self._max_attempts,
                    exc.message,
                )
                continue

            logger.info("Judge evaluation completed attempts=%d", attempt)
            return JudgeResult(
                scores={model: parsed.scores[label_of[model]] for model in responses},
                reasoning=parsed.reasoning,
                label_map=label_map,
                raw=raw,
                attempts=attempt,
                truncated_models=[label_map[label] for label in truncated_labels],
            )

        raise InvalidJudgeResponseError(
            f"Judge '{judge_model}' returned no valid output after {self._max_attempts} "
            f"attempts: {errors[-1]}",
            {"raw": raw, "attempts": self._max_attempts, "errors": errors},
        )

    def _response_budget_chars(self, prompt: str, labels: list[str], nonce: str) -> int:
        s = self._settings
        available_tokens = s.judge_num_ctx - s.judge_max_tokens - CONTEXT_SAFETY_TOKENS
        fixed_chars = (
            len(self._system_prompt)
            + len(build_user_prompt(prompt, {label: "" for label in labels}, nonce))
            + FEEDBACK_RESERVE_CHARS
        )
        return available_tokens * CHARS_PER_TOKEN - fixed_chars

    @staticmethod
    def _feedback(errors: list[str]) -> str:
        if not errors:
            return ""
        return (
            f"\n\nYour previous reply was rejected: {errors[-1]}. "
            "Reply again with ONLY a valid JSON object that scores every label."
        )
