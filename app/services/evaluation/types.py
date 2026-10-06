"""Types shared by the evaluation pipeline."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class RunStatus(StrEnum):
    OK = "ok"
    TIMEOUT = "timeout"
    EMPTY = "empty"
    ERROR = "error"  # connection problem, model not installed, malformed response, ...


class ModelRun(BaseModel):
    """Outcome of sending the prompt to one model. Failures are data, not exceptions."""

    model_config = ConfigDict(frozen=True)

    model: str
    status: RunStatus
    text: str | None = None
    error: str | None = None
    error_code: str | None = None
    # Wall-clock time of the call (for failures: time until the failure).
    latency_ms: float
    load_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    tokens_per_sec: float | None = None

    @property
    def ok(self) -> bool:
        return self.status is RunStatus.OK


class ValidatedRequest(BaseModel):
    """A request that passed validation, with canonical model names."""

    prompt: str
    models: list[str]
    judge: str
    judge_is_participant: bool
