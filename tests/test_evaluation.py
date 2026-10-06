import time

import pytest

from app.core.config import Settings
from app.core.errors import (
    DuplicateModelsError,
    EmptyPromptError,
    EmptyResponseError,
    GenerationTimeoutError,
    InvalidRequestError,
    ModelNotInstalledError,
    NotEnoughModelsError,
    PromptTooLongError,
    ProviderUnavailableError,
    TooManyModelsError,
)
from app.services.evaluation.evaluation_service import EvaluationService
from app.services.evaluation.types import RunStatus
from app.services.llm.types import GenerationRequest, GenerationResult
from tests.conftest import FakeProvider


def make_service(provider: FakeProvider, **settings) -> EvaluationService:
    grace = settings.pop("timeout_grace", 5.0)
    return EvaluationService(provider, Settings(_env_file=None, **settings), timeout_grace=grace)


class TrackingProvider(FakeProvider):
    """Records how many generations run at the same time."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.active = 0
        self.peak = 0

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            return await super().generate(request)
        finally:
            self.active -= 1


# ------------------------------------------------------------------ collection
async def test_models_run_in_parallel() -> None:
    provider = FakeProvider(models=["a", "b", "c", "d"], delay=0.2)
    service = make_service(provider, max_concurrency=4)

    start = time.perf_counter()
    runs = await service.collect_responses("hi", ["a", "b", "c", "d"])
    elapsed = time.perf_counter() - start

    assert all(r.ok for r in runs)
    assert elapsed < 0.5  # sequential execution would take >= 0.8s


async def test_results_keep_input_order_and_carry_data() -> None:
    provider = FakeProvider(delay=0.01)
    runs = await make_service(provider).collect_responses("hi", ["c", "a", "b"])
    assert [r.model for r in runs] == ["c", "a", "b"]
    assert runs[0].text == "answer from c"
    assert runs[0].status is RunStatus.OK and runs[0].error is None


async def test_participant_request_uses_configured_limits() -> None:
    provider = FakeProvider()
    service = make_service(provider, participant_max_tokens=321, request_timeout=42)
    await service.collect_responses("hello", ["a", "b"])
    assert {c.max_tokens for c in provider.calls} == {321}
    assert {c.timeout for c in provider.calls} == {42}
    assert {c.prompt for c in provider.calls} == {"hello"}


async def test_one_failing_model_does_not_break_the_round() -> None:
    provider = FakeProvider(responses={"b": ProviderUnavailableError("connection refused")})
    runs = await make_service(provider).collect_responses("hi", ["a", "b", "c"])

    by_model = {r.model: r for r in runs}
    assert by_model["a"].ok and by_model["c"].ok
    assert by_model["b"].status is RunStatus.ERROR
    assert by_model["b"].error_code == "provider_unavailable"
    assert by_model["b"].text is None


@pytest.mark.parametrize(
    ("exc", "status", "code"),
    [
        (GenerationTimeoutError("timed out"), RunStatus.TIMEOUT, "generation_timeout"),
        (EmptyResponseError("empty"), RunStatus.EMPTY, "empty_response"),
        (ModelNotInstalledError("missing"), RunStatus.ERROR, "model_not_installed"),
        (RuntimeError("bug"), RunStatus.ERROR, "unexpected_error"),
    ],
)
async def test_failure_classification(exc: Exception, status: RunStatus, code: str) -> None:
    provider = FakeProvider(responses={"a": exc})
    runs = await make_service(provider).collect_responses("hi", ["a", "b"])
    assert runs[0].status is status and runs[0].error_code == code
    assert runs[0].error
    assert runs[1].ok


async def test_all_models_failing_returns_failures_without_raising() -> None:
    provider = FakeProvider(
        responses={"a": RuntimeError("x"), "b": ProviderUnavailableError("down")}
    )
    runs = await make_service(provider).collect_responses("hi", ["a", "b"])
    assert [r.ok for r in runs] == [False, False]


async def test_backstop_timeout_catches_a_hanging_provider() -> None:
    provider = FakeProvider(delay=30)  # never finishes within the test
    service = make_service(provider, request_timeout=0.1, timeout_grace=0.05)

    start = time.perf_counter()
    runs = await service.collect_responses("hi", ["a", "b"])

    assert time.perf_counter() - start < 2
    assert all(r.status is RunStatus.TIMEOUT for r in runs)
    assert all(r.latency_ms >= 100 for r in runs)


async def test_concurrency_is_limited_by_semaphore() -> None:
    provider = TrackingProvider(delay=0.05)
    service = make_service(provider, max_concurrency=2)
    runs = await service.collect_responses("hi", ["a", "b", "c", "d", "e"])
    assert all(r.ok for r in runs)
    assert provider.peak == 2


# ------------------------------------------------------------------ validation
async def validate(provider=None, prompt="hello", models=("alpha", "beta"), judge="gamma", **kw):
    service = make_service(provider or FakeProvider(), **kw)
    return await service.validate_request(prompt, list(models), judge)


async def test_valid_request_is_normalized() -> None:
    req = await validate(prompt="  hello  ", models=[" alpha ", "beta:latest"], judge="gamma")
    assert req.prompt == "hello"
    assert req.models == ["alpha:latest", "beta:latest"]
    assert req.judge == "gamma:latest"
    assert req.judge_is_participant is False


async def test_judge_is_participant_flag() -> None:
    req = await validate(models=["alpha", "beta"], judge="beta:latest")
    assert req.judge_is_participant is True


@pytest.mark.parametrize("prompt", ["", "   ", "\n\t"])
async def test_empty_prompt_rejected(prompt: str) -> None:
    with pytest.raises(EmptyPromptError):
        await validate(prompt=prompt)


async def test_prompt_too_long_rejected() -> None:
    with pytest.raises(PromptTooLongError):
        await validate(prompt="x" * 11, max_prompt_chars=10)


async def test_too_few_models_rejected() -> None:
    with pytest.raises(NotEnoughModelsError):
        await validate(models=["alpha"])


async def test_too_many_models_rejected() -> None:
    with pytest.raises(TooManyModelsError) as exc:
        await validate(models=["a", "b", "c"], max_models=2)
    assert exc.value.details == {"got": 3, "max": 2}


async def test_duplicates_rejected_including_implicit_latest_tag() -> None:
    with pytest.raises(DuplicateModelsError) as exc:
        await validate(models=["alpha", "alpha:latest", "beta"])
    assert exc.value.details["duplicates"] == ["alpha:latest"]


@pytest.mark.parametrize(("models", "judge"), [(["alpha", ""], "gamma"), (["alpha", "beta"], " ")])
async def test_blank_names_rejected(models: list[str], judge: str) -> None:
    with pytest.raises(InvalidRequestError):
        await validate(models=models, judge=judge)


async def test_not_installed_models_reported_including_judge() -> None:
    with pytest.raises(ModelNotInstalledError) as exc:
        await validate(models=["alpha", "nope"], judge="ghost")
    assert exc.value.details["missing"] == ["nope:latest", "ghost:latest"]


async def test_validation_surfaces_provider_outage() -> None:
    provider = FakeProvider(list_error=ProviderUnavailableError("Cannot connect"))
    with pytest.raises(ProviderUnavailableError):
        await validate(provider=provider)
