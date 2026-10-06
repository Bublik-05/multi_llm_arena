"""Provider-agnostic request/response types for LLM backends."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class GenerationRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    prompt: str
    system: str | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, gt=0)
    num_ctx: int | None = Field(default=None, gt=0)
    # JSON Schema for structured output (used by the judge).
    json_schema: dict[str, Any] | None = None
    # None = leave the model's default. True/False toggles "thinking" where supported.
    think: bool | None = None
    # Per-request timeout in seconds; falls back to the provider default.
    timeout: float | None = Field(default=None, gt=0)


class GenerationResult(BaseModel):
    model: str
    text: str
    # Wall-clock time of the whole call, measured by us (includes model load time).
    latency_ms: float
    # Provider-reported model load time. A large value means a cold start.
    load_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    # Generation speed excluding load and prompt processing; comparable across runs.
    tokens_per_sec: float | None = None
    done_reason: str | None = None


class ModelInfo(BaseModel):
    name: str
    provider: str
    size_bytes: int | None = None
    family: str | None = None
    parameter_size: str | None = None
    quantization: str | None = None
    context_length: int | None = None
    capabilities: list[str] = Field(default_factory=list)


class ProviderHealth(BaseModel):
    available: bool
    provider: str
    version: str | None = None
    detail: str | None = None
