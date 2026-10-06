from fastapi import APIRouter

from app import __version__
from app.api.deps import ProviderDep
from app.schemas.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(provider: ProviderDep) -> HealthResponse:
    """Always 200 while the app is alive; status is 'degraded' if the LLM backend is down."""
    provider_health = await provider.health_check()
    return HealthResponse(
        status="ok" if provider_health.available else "degraded",
        version=__version__,
        provider=provider_health,
    )
