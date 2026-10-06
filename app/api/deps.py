from typing import Annotated

from fastapi import Depends, Request

from app.core.errors import ProviderUnavailableError
from app.services.llm.base import LLMProvider


def get_provider(request: Request) -> LLMProvider:
    provider = getattr(request.app.state, "provider", None)
    if provider is None:
        raise ProviderUnavailableError("LLM provider is not initialised")
    return provider


ProviderDep = Annotated[LLMProvider, Depends(get_provider)]
