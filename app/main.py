import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.routes import health, models
from app.core.config import get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import setup_logging
from app.services.llm.base import LLMProvider
from app.services.llm.factory import create_provider

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    setup_logging(settings.log_level)
    logger.info("Starting Multi-LLM Arena v%s", __version__)
    if app.state.provider is None:
        app.state.provider = create_provider(settings)

    provider_health = await app.state.provider.health_check()
    if provider_health.available:
        logger.info(
            "%s reachable at %s (version=%s)",
            provider_health.provider,
            settings.ollama_base_url,
            provider_health.version,
        )
    else:
        logger.warning("%s is not reachable: %s", provider_health.provider, provider_health.detail)
    # Later phases: init DB, mark interrupted evaluations.
    yield
    await app.state.provider.aclose()
    logger.info("Shutting down")


def create_app(provider: LLMProvider | None = None) -> FastAPI:
    """`provider` lets tests inject a fake; in production it is created in lifespan."""
    settings = get_settings()
    app = FastAPI(
        title="Multi-LLM Evaluation Arena",
        version=__version__,
        description="Compare local LLMs with an anonymous judge and an Elo leaderboard.",
        lifespan=lifespan,
    )
    app.state.provider = provider
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(models.router)
    return app


app = create_app()
