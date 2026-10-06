from pydantic import BaseModel

from app.services.llm.types import ModelInfo


class ModelListResponse(BaseModel):
    provider: str
    models: list[ModelInfo]
