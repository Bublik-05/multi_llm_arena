"""Orchestrates one evaluation round: validate -> collect responses (-> judge -> rank later)."""

import asyncio
import logging
import time
from collections.abc import Sequence

from app.core.config import Settings
from app.core.errors import (
    ArenaError,
    DuplicateModelsError,
    EmptyPromptError,
    EmptyResponseError,
    GenerationTimeoutError,
    InvalidRequestError,
    ModelNotInstalledError,
    NotEnoughModelsError,
    PromptTooLongError,
    TooManyModelsError,
)
from app.services.evaluation.types import ModelRun, RunStatus, ValidatedRequest
from app.services.llm.base import LLMProvider
from app.services.llm.types import GenerationRequest

logger = logging.getLogger(__name__)


class EvaluationService:
    def __init__(
        self, provider: LLMProvider, settings: Settings, *, timeout_grace: float = 5.0
    ) -> None:
        self._provider = provider
        self._settings = settings
        # One semaphore per service instance: bounds concurrent generations across all
        # evaluations, not just within one round.
        self._semaphore = asyncio.Semaphore(settings.max_concurrency)
        # The provider enforces request_timeout itself; this is a backstop so a misbehaving
        # provider can never hang a round.
        self._timeout_grace = timeout_grace

    # ------------------------------------------------------------------ validation
    async def validate_request(
        self, prompt: str, models: Sequence[str], judge: str
    ) -> ValidatedRequest:
        """Check a user request and return it with canonical model names.

        Raises an InvalidRequestError subclass (HTTP 422) on the first problem found.
        The installed-models check goes last because it needs a network call.
        """
        s = self._settings
        prompt = prompt.strip()
        if not prompt:
            raise EmptyPromptError("Prompt must not be empty")
        if len(prompt) > s.max_prompt_chars:
            raise PromptTooLongError(
                f"Prompt is too long ({len(prompt)} chars, max {s.max_prompt_chars})",
                {"length": len(prompt), "max": s.max_prompt_chars},
            )

        names = [m.strip() for m in models]
        judge = judge.strip()
        if not judge or any(not n for n in names):
            raise InvalidRequestError("Model and judge names must not be empty")
        if len(names) < 2:
            raise NotEnoughModelsError("Select at least 2 models to compare", {"got": len(names)})
        if len(names) > s.max_models:
            raise TooManyModelsError(
                f"Too many models selected ({len(names)}, max {s.max_models})",
                {"got": len(names), "max": s.max_models},
            )

        # Compare canonical names so "llama3.2" and "llama3.2:latest" count as duplicates.
        canonical = [self._provider.normalize_model_name(n) for n in names]
        duplicates = sorted({n for n in canonical if canonical.count(n) > 1})
        if duplicates:
            raise DuplicateModelsError(
                f"Duplicate models selected: {', '.join(duplicates)}", {"duplicates": duplicates}
            )
        canonical_judge = self._provider.normalize_model_name(judge)

        missing = await self._provider.find_missing_models([*canonical, canonical_judge])
        if missing:
            raise ModelNotInstalledError(
                f"Models not installed: {', '.join(missing)}", {"missing": missing}
            )

        return ValidatedRequest(
            prompt=prompt,
            models=canonical,
            judge=canonical_judge,
            judge_is_participant=canonical_judge in canonical,
        )

    # ------------------------------------------------------------------ collection
    async def collect_responses(self, prompt: str, models: Sequence[str]) -> list[ModelRun]:
        """Send the prompt to every model concurrently.

        Never raises because of a model failure: each failure becomes a ModelRun with a
        non-OK status. Results are returned in the same order as `models`.
        """
        logger.info("Starting evaluation models=%s", list(models))
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(self._run_one(prompt, m)) for m in models]
        runs = [t.result() for t in tasks]
        failed = [r for r in runs if not r.ok]
        logger.info("Responses collected ok=%d failed=%d", len(runs) - len(failed), len(failed))
        return runs

    async def _run_one(self, prompt: str, model: str) -> ModelRun:
        async with self._semaphore:
            start = time.perf_counter()
            try:
                request = GenerationRequest(
                    model=model,
                    prompt=prompt,
                    max_tokens=self._settings.participant_max_tokens,
                    timeout=self._settings.request_timeout,
                )
                async with asyncio.timeout(self._settings.request_timeout + self._timeout_grace):
                    result = await self._provider.generate(request)
            except TimeoutError:
                return self._failure(
                    model, start, RunStatus.TIMEOUT, "generation_timeout", "Generation timed out"
                )
            except GenerationTimeoutError as exc:
                return self._failure(model, start, RunStatus.TIMEOUT, exc.code, exc.message)
            except EmptyResponseError as exc:
                return self._failure(model, start, RunStatus.EMPTY, exc.code, exc.message)
            except ArenaError as exc:
                return self._failure(model, start, RunStatus.ERROR, exc.code, exc.message)
            except Exception as exc:  # a bug in one provider call must not kill the round
                logger.exception("Unexpected error while running model=%s", model)
                return self._failure(
                    model,
                    start,
                    RunStatus.ERROR,
                    "unexpected_error",
                    f"Unexpected error: {type(exc).__name__}",
                )

        return ModelRun(
            model=model,
            status=RunStatus.OK,
            text=result.text,
            latency_ms=result.latency_ms,
            load_ms=result.load_ms,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            tokens_per_sec=result.tokens_per_sec,
        )

    @staticmethod
    def _failure(model: str, start: float, status: RunStatus, code: str, message: str) -> ModelRun:
        logger.warning("Model failed model=%s status=%s error=%s", model, status.value, message)
        return ModelRun(
            model=model,
            status=status,
            error=message,
            error_code=code,
            latency_ms=(time.perf_counter() - start) * 1000,
        )
