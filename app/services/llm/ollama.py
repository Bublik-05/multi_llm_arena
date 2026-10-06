"""Ollama provider. Talks to the REST API directly via httpx (no SDK dependency)."""

import logging
import time
from typing import Any

import httpx

from app.core.errors import (
    EmptyResponseError,
    GenerationTimeoutError,
    MalformedResponseError,
    ModelNotInstalledError,
    ProviderError,
    ProviderUnavailableError,
)
from app.services.llm.base import LLMProvider
from app.services.llm.types import GenerationRequest, GenerationResult, ModelInfo, ProviderHealth
from app.services.llm.utils import strip_think_blocks

logger = logging.getLogger(__name__)

_NS_PER_MS = 1_000_000
_NS_PER_S = 1_000_000_000


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(
        self,
        base_url: str,
        timeout: float = 120.0,
        connect_timeout: float = 5.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._timeout = timeout
        self._connect_timeout = connect_timeout
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(base_url=base_url.rstrip("/"))

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def normalize_model_name(self, name: str) -> str:
        # Ollama resolves "llama3.2" to "llama3.2:latest".
        name = name.strip()
        return name if ":" in name else f"{name}:latest"

    # ------------------------------------------------------------------ generation
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        payload = self._build_chat_payload(request)
        logger.info("Sending request to model=%s", request.model)
        start = time.perf_counter()

        resp = await self._request(
            "POST", "/api/chat", json=payload, timeout_s=request.timeout, model=request.model
        )
        if resp.status_code == 400 and "think" in payload and "think" in self._error_text(resp):
            # Model does not support the think option: retry once without it.
            logger.info("model=%s does not support 'think'; retrying without it", request.model)
            payload.pop("think")
            resp = await self._request(
                "POST", "/api/chat", json=payload, timeout_s=request.timeout, model=request.model
            )

        latency_ms = (time.perf_counter() - start) * 1000
        self._raise_for_status(resp, request.model)
        result = self._parse_chat(resp, request.model, latency_ms)
        logger.info(
            "Model response received model=%s latency=%.2fs", request.model, latency_ms / 1000
        )
        return result

    @staticmethod
    def _build_chat_payload(request: GenerationRequest) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})

        options: dict[str, Any] = {}
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        if request.num_ctx is not None:
            options["num_ctx"] = request.num_ctx

        payload: dict[str, Any] = {"model": request.model, "messages": messages, "stream": False}
        if options:
            payload["options"] = options
        if request.json_schema is not None:
            payload["format"] = request.json_schema
        if request.think is not None:
            payload["think"] = request.think
        return payload

    def _parse_chat(self, resp: httpx.Response, model: str, latency_ms: float) -> GenerationResult:
        data = self._json(resp)
        message = data.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise MalformedResponseError(
                "Unexpected response shape from Ollama /api/chat", {"model": model}
            )

        done_reason = data.get("done_reason")
        text = strip_think_blocks(message["content"]).strip()
        if not text:
            hint = ""
            if done_reason == "length":
                hint = (
                    " (token limit reached; reasoning models can spend the whole budget thinking)"
                )
            raise EmptyResponseError(
                f"Model '{model}' returned an empty response{hint}",
                {"model": model, "done_reason": done_reason},
            )

        eval_count, eval_duration = data.get("eval_count"), data.get("eval_duration")
        tokens_per_sec = None
        if _is_number(eval_count) and _is_number(eval_duration) and eval_duration > 0:
            tokens_per_sec = eval_count / (eval_duration / _NS_PER_S)
        load = data.get("load_duration")
        prompt_tokens = data.get("prompt_eval_count")

        return GenerationResult(
            model=model,
            text=text,
            latency_ms=latency_ms,
            load_ms=load / _NS_PER_MS if _is_number(load) else None,
            prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
            completion_tokens=eval_count if isinstance(eval_count, int) else None,
            tokens_per_sec=tokens_per_sec,
            done_reason=done_reason if isinstance(done_reason, str) else None,
        )

    # ------------------------------------------------------------------ models / health
    async def health_check(self) -> ProviderHealth:
        try:
            resp = await self._request("GET", "/api/version", timeout_s=self._connect_timeout)
            self._raise_for_status(resp)
            version = self._json(resp).get("version")
        except ProviderError as exc:
            return ProviderHealth(available=False, provider=self.name, detail=exc.message)
        return ProviderHealth(
            available=True,
            provider=self.name,
            version=version if isinstance(version, str) else None,
        )

    async def list_models(self) -> list[ModelInfo]:
        resp = await self._request("GET", "/api/tags")
        self._raise_for_status(resp)
        models: list[ModelInfo] = []
        for item in self._json(resp).get("models") or []:
            name = item.get("name") or item.get("model")
            if not isinstance(name, str):
                continue
            details = item.get("details") or {}
            size = item.get("size")
            models.append(
                ModelInfo(
                    name=name,
                    provider=self.name,
                    size_bytes=size if isinstance(size, int) else None,
                    family=details.get("family"),
                    parameter_size=details.get("parameter_size"),
                    quantization=details.get("quantization_level"),
                )
            )
        return models

    async def get_model_info(self, model: str) -> ModelInfo:
        resp = await self._request("POST", "/api/show", json={"model": model}, model=model)
        self._raise_for_status(resp, model)
        data = self._json(resp)
        details = data.get("details") or {}
        model_info = data.get("model_info") or {}
        arch = model_info.get("general.architecture")
        context_length = model_info.get(f"{arch}.context_length") if arch else None
        capabilities = data.get("capabilities")
        return ModelInfo(
            name=model,
            provider=self.name,
            family=details.get("family"),
            parameter_size=details.get("parameter_size"),
            quantization=details.get("quantization_level"),
            context_length=context_length if isinstance(context_length, int) else None,
            capabilities=[c for c in capabilities if isinstance(c, str)]
            if isinstance(capabilities, list)
            else [],
        )

    # ------------------------------------------------------------------ HTTP plumbing
    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        timeout_s: float | None = None,
        model: str | None = None,
    ) -> httpx.Response:
        limit = timeout_s or self._timeout
        try:
            return await self._client.request(
                method,
                path,
                json=json,
                timeout=httpx.Timeout(limit, connect=min(self._connect_timeout, limit)),
            )
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise ProviderUnavailableError(
                f"Cannot connect to Ollama at {self._client.base_url}. Is it running?"
            ) from exc
        except httpx.TimeoutException as exc:
            raise GenerationTimeoutError(
                f"Ollama did not respond within {limit:.0f}s", {"model": model}
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"HTTP error while talking to Ollama: {type(exc).__name__}", {"model": model}
            ) from exc

    @staticmethod
    def _error_text(resp: httpx.Response) -> str:
        try:
            body = resp.json()
            if isinstance(body, dict) and isinstance(body.get("error"), str):
                return body["error"].lower()
        except ValueError:
            pass
        return resp.text[:200].lower()

    def _raise_for_status(self, resp: httpx.Response, model: str | None = None) -> None:
        if resp.is_success:
            return
        text = self._error_text(resp)
        if resp.status_code == 404 and model and "not found" in text:
            raise ModelNotInstalledError(
                f"Model '{model}' is not installed in Ollama. Run: ollama pull {model}",
                {"model": model},
            )
        raise ProviderError(
            f"Ollama returned HTTP {resp.status_code}: {text}",
            {"model": model, "status": resp.status_code},
        )

    @staticmethod
    def _json(resp: httpx.Response) -> dict[str, Any]:
        try:
            data = resp.json()
        except ValueError as exc:
            raise MalformedResponseError("Ollama returned invalid JSON") from exc
        if not isinstance(data, dict):
            raise MalformedResponseError("Ollama returned an unexpected JSON structure")
        return data
