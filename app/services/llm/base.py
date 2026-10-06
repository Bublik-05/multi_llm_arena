"""LLMProvider: the only interface the rest of the app uses to talk to a model backend."""

from abc import ABC, abstractmethod
from collections.abc import Iterable

from app.services.llm.types import GenerationRequest, GenerationResult, ModelInfo, ProviderHealth


class LLMProvider(ABC):
    name: str

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Run one completion. Raises a ProviderError subclass on failure."""

    @abstractmethod
    async def health_check(self) -> ProviderHealth:
        """Never raises: reports availability in the returned object."""

    @abstractmethod
    async def list_models(self) -> list[ModelInfo]:
        """Models that can be used right now (e.g. installed in Ollama)."""

    @abstractmethod
    async def get_model_info(self, model: str) -> ModelInfo:
        """Detailed info for one model. Raises ModelNotInstalledError if unknown."""

    async def aclose(self) -> None:
        """Release network resources. Default: nothing to release."""
        return None

    def normalize_model_name(self, name: str) -> str:
        """Canonical form used to compare user-supplied names with installed ones."""
        return name.strip()

    async def find_missing_models(self, names: Iterable[str]) -> list[str]:
        installed = {self.normalize_model_name(m.name) for m in await self.list_models()}
        return [n for n in names if self.normalize_model_name(n) not in installed]
