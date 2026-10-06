from fastapi import APIRouter

from app.api.deps import ProviderDep
from app.schemas.models import ModelListResponse

router = APIRouter(tags=["models"])


@router.get("/models", response_model=ModelListResponse)
async def list_models(provider: ProviderDep) -> ModelListResponse:
    """Models actually available in the backend (read live from Ollama, nothing hardcoded)."""
    return ModelListResponse(provider=provider.name, models=await provider.list_models())
