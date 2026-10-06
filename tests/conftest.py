import asyncio
from collections.abc import Iterable

from app.services.llm.base import LLMProvider
from app.services.llm.types import GenerationRequest, GenerationResult, ModelInfo, ProviderHealth


class FakeProvider(LLMProvider):
    """In-memory provider for tests.

    `responses` maps model name -> text or an Exception to raise. Models not listed return
    "answer from <model>".
    """

    name = "fake"

    def __init__(
        self,
        models: Iterable[str] = ("alpha", "beta", "gamma"),
        responses: dict[str, str | Exception] | None = None,
        delay: float = 0.0,
        healthy: bool = True,
        list_error: Exception | None = None,
    ) -> None:
        self.models = list(models)
        self.responses = responses or {}
        self.delay = delay
        self.healthy = healthy
        self.list_error = list_error
        self.calls: list[GenerationRequest] = []

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls.append(request)
        if self.delay:
            await asyncio.sleep(self.delay)
        outcome = self.responses.get(request.model, f"answer from {request.model}")
        if isinstance(outcome, Exception):
            raise outcome
        return GenerationResult(model=request.model, text=outcome, latency_ms=self.delay * 1000)

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(
            available=self.healthy,
            provider=self.name,
            version="fake-1" if self.healthy else None,
            detail=None if self.healthy else "down",
        )

    async def list_models(self) -> list[ModelInfo]:
        if self.list_error:
            raise self.list_error
        return [ModelInfo(name=m, provider=self.name) for m in self.models]

    async def get_model_info(self, model: str) -> ModelInfo:
        return ModelInfo(name=model, provider=self.name)

    def normalize_model_name(self, name: str) -> str:
        name = name.strip()
        return name if ":" in name else f"{name}:latest"
