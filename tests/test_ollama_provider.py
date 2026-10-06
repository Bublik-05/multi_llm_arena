import json

import httpx
import pytest

from app.core.errors import (
    EmptyResponseError,
    GenerationTimeoutError,
    MalformedResponseError,
    ModelNotInstalledError,
    ProviderError,
    ProviderUnavailableError,
)
from app.services.llm.ollama import OllamaProvider
from app.services.llm.types import GenerationRequest


def make_provider(handler) -> OllamaProvider:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://ollama.test"
    )
    return OllamaProvider("http://ollama.test", timeout=5, client=client)


def chat_ok(content: str = "hello", **extra) -> dict:
    body = {
        "model": "m",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "done_reason": "stop",
        "load_duration": 2_000_000_000,
        "prompt_eval_count": 12,
        "eval_count": 50,
        "eval_duration": 2_500_000_000,
    }
    body.update(extra)
    return body


REQ = GenerationRequest(model="m", prompt="hi")


async def test_generate_success_and_payload_mapping() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=chat_ok("the answer"))

    provider = make_provider(handler)
    result = await provider.generate(
        GenerationRequest(
            model="m",
            prompt="hi",
            system="be brief",
            temperature=0.2,
            max_tokens=100,
            num_ctx=8192,
            json_schema={"type": "object"},
            think=False,
        )
    )

    assert seen["stream"] is False
    assert seen["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ]
    assert seen["options"] == {"temperature": 0.2, "num_predict": 100, "num_ctx": 8192}
    assert seen["format"] == {"type": "object"}
    assert seen["think"] is False

    assert result.text == "the answer"
    assert result.load_ms == pytest.approx(2000)
    assert result.prompt_tokens == 12
    assert result.completion_tokens == 50
    assert result.tokens_per_sec == pytest.approx(20.0)
    assert result.latency_ms > 0


async def test_optional_fields_are_omitted_from_payload() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=chat_ok())

    await make_provider(handler).generate(REQ)
    assert "options" not in seen and "format" not in seen and "think" not in seen


async def test_think_blocks_are_stripped() -> None:
    body = chat_ok("<think>hmm\nlet me see</think>\n\nFinal answer")
    provider = make_provider(lambda r: httpx.Response(200, json=body))
    assert (await provider.generate(REQ)).text == "Final answer"


async def test_unterminated_think_block_gives_empty_response() -> None:
    body = chat_ok("<think>still thinking...", done_reason="length")
    provider = make_provider(lambda r: httpx.Response(200, json=body))
    with pytest.raises(EmptyResponseError) as exc:
        await provider.generate(REQ)
    assert "token limit" in exc.value.message


async def test_empty_response() -> None:
    provider = make_provider(lambda r: httpx.Response(200, json=chat_ok("   ")))
    with pytest.raises(EmptyResponseError):
        await provider.generate(REQ)


async def test_model_not_installed() -> None:
    provider = make_provider(lambda r: httpx.Response(404, json={"error": "model 'm' not found"}))
    with pytest.raises(ModelNotInstalledError) as exc:
        await provider.generate(REQ)
    assert "ollama pull m" in exc.value.message


async def test_connection_error_maps_to_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ProviderUnavailableError):
        await make_provider(handler).generate(REQ)


async def test_read_timeout_maps_to_generation_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(GenerationTimeoutError):
        await make_provider(handler).generate(REQ)


async def test_connect_timeout_maps_to_unavailable_not_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("no route", request=request)

    with pytest.raises(ProviderUnavailableError):
        await make_provider(handler).generate(REQ)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, content=b"<html>not json</html>"),
        httpx.Response(200, json={"done": True}),
        httpx.Response(200, json={"message": {"content": 123}}),
        httpx.Response(200, json=["a", "list"]),
    ],
)
async def test_malformed_responses(response: httpx.Response) -> None:
    with pytest.raises(MalformedResponseError):
        await make_provider(lambda r: response).generate(REQ)


async def test_server_error_message_is_surfaced() -> None:
    provider = make_provider(
        lambda r: httpx.Response(500, json={"error": "model requires more system memory"})
    )
    with pytest.raises(ProviderError) as exc:
        await provider.generate(REQ)
    assert "more system memory" in exc.value.message
    assert exc.value.details["status"] == 500


async def test_retries_without_think_when_unsupported() -> None:
    payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        payloads.append(payload)
        if "think" in payload:
            return httpx.Response(400, json={"error": '"m" does not support thinking'})
        return httpx.Response(200, json=chat_ok("ok"))

    result = await make_provider(handler).generate(
        GenerationRequest(model="m", prompt="hi", think=False)
    )
    assert result.text == "ok"
    assert len(payloads) == 2 and "think" not in payloads[1]


async def test_other_400_errors_are_not_retried() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, json={"error": "bad options"})

    with pytest.raises(ProviderError):
        await make_provider(handler).generate(
            GenerationRequest(model="m", prompt="hi", think=False)
        )
    assert calls == 1


async def test_list_models() -> None:
    body = {
        "models": [
            {
                "name": "llama3.2:latest",
                "size": 2_000_000_000,
                "details": {
                    "family": "llama",
                    "parameter_size": "3.2B",
                    "quantization_level": "Q4_K_M",
                },
            },
            {"model": "qwen3:8b"},
            {"size": 1},  # no name: skipped
        ]
    }
    provider = make_provider(lambda r: httpx.Response(200, json=body))
    models = await provider.list_models()
    assert [m.name for m in models] == ["llama3.2:latest", "qwen3:8b"]
    assert models[0].parameter_size == "3.2B" and models[0].quantization == "Q4_K_M"


async def test_find_missing_models_normalizes_latest_tag() -> None:
    body = {"models": [{"name": "llama3.2:latest"}, {"name": "qwen3:8b"}]}
    provider = make_provider(lambda r: httpx.Response(200, json=body))
    missing = await provider.find_missing_models(["llama3.2", "qwen3:8b", "qwen3", "gemma3"])
    assert missing == ["qwen3", "gemma3"]


async def test_get_model_info() -> None:
    body = {
        "details": {"family": "llama", "parameter_size": "3.2B", "quantization_level": "Q4_K_M"},
        "model_info": {"general.architecture": "llama", "llama.context_length": 131072},
        "capabilities": ["completion", "tools"],
    }
    provider = make_provider(lambda r: httpx.Response(200, json=body))
    info = await provider.get_model_info("llama3.2")
    assert info.context_length == 131072
    assert info.capabilities == ["completion", "tools"]


async def test_get_model_info_not_installed() -> None:
    provider = make_provider(lambda r: httpx.Response(404, json={"error": "model 'x' not found"}))
    with pytest.raises(ModelNotInstalledError):
        await provider.get_model_info("x")


async def test_health_check_ok() -> None:
    provider = make_provider(lambda r: httpx.Response(200, json={"version": "0.12.3"}))
    health = await provider.health_check()
    assert health.available and health.version == "0.12.3"


async def test_health_check_never_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    health = await make_provider(handler).health_check()
    assert not health.available
    assert "Cannot connect" in (health.detail or "")
