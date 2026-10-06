from app.core.config import Settings
from app.services.llm.base import LLMProvider
from app.services.llm.ollama import OllamaProvider


def create_provider(settings: Settings) -> LLMProvider:
    """Only Ollama for now; other providers would be selected here."""
    return OllamaProvider(
        base_url=settings.ollama_base_url,
        timeout=settings.request_timeout,
        connect_timeout=settings.connect_timeout,
    )
