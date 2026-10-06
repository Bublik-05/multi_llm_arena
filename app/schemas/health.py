from typing import Literal

from pydantic import BaseModel

from app.services.llm.types import ProviderHealth


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    provider: ProviderHealth
