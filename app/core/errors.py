"""Domain exceptions and their HTTP mapping.

Services raise ArenaError subclasses; the API layer turns them into JSON:
    {"error": {"code": "...", "message": "...", "details": {...}}}
"""

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class ArenaError(Exception):
    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


# --- Request validation (HTTP 422) -------------------------------------------------------
class InvalidRequestError(ArenaError):
    status_code = 422
    code = "invalid_request"


class EmptyPromptError(InvalidRequestError):
    code = "empty_prompt"


class PromptTooLongError(InvalidRequestError):
    code = "prompt_too_long"


class DuplicateModelsError(InvalidRequestError):
    code = "duplicate_models"


class TooManyModelsError(InvalidRequestError):
    code = "too_many_models"


class NotEnoughModelsError(InvalidRequestError):
    code = "not_enough_models"


class ModelNotInstalledError(InvalidRequestError):
    code = "model_not_installed"


# --- Not found (HTTP 404) ----------------------------------------------------------------
class NotFoundError(ArenaError):
    status_code = 404
    code = "not_found"


# --- Upstream LLM failures ---------------------------------------------------------------
class ProviderError(ArenaError):
    """Base class for failures while talking to an LLM backend."""

    status_code = 502
    code = "provider_error"


class ProviderUnavailableError(ProviderError):
    status_code = 503
    code = "provider_unavailable"


class GenerationTimeoutError(ProviderError):
    status_code = 504
    code = "generation_timeout"


class EmptyResponseError(ProviderError):
    code = "empty_response"


class MalformedResponseError(ProviderError):
    code = "malformed_response"


class InvalidJudgeResponseError(ProviderError):
    code = "invalid_judge_response"


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ArenaError)
    async def _arena_error(_: Request, exc: ArenaError) -> JSONResponse:
        if exc.status_code >= 500:
            logger.warning("%s: %s", exc.code, exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message, "details": exc.details}},
        )

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "Internal server error",
                    "details": {},
                }
            },
        )
