import httpx

from app.core.errors import ProviderUnavailableError
from app.main import create_app
from tests.conftest import FakeProvider


def _client(provider: FakeProvider) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(provider)), base_url="http://test"
    )


async def test_health_ok() -> None:
    async with _client(FakeProvider()) as client:
        resp = await client.get("/health")
    body = resp.json()
    assert resp.status_code == 200
    assert body["status"] == "ok"
    assert body["provider"]["available"] is True


async def test_health_degraded_when_provider_down() -> None:
    async with _client(FakeProvider(healthy=False)) as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "degraded"
    assert resp.json()["provider"]["detail"] == "down"


async def test_models_lists_provider_models() -> None:
    async with _client(FakeProvider(models=["a:latest", "b:8b"])) as client:
        resp = await client.get("/models")
    assert resp.status_code == 200
    assert [m["name"] for m in resp.json()["models"]] == ["a:latest", "b:8b"]
    assert resp.json()["provider"] == "fake"


async def test_models_returns_503_when_ollama_unavailable() -> None:
    provider = FakeProvider(list_error=ProviderUnavailableError("Cannot connect to Ollama"))
    async with _client(provider) as client:
        resp = await client.get("/models")
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "provider_unavailable"


async def test_arena_error_is_serialized() -> None:
    from app.core.errors import DuplicateModelsError

    app = create_app(FakeProvider())

    @app.get("/_boom")
    async def boom() -> None:
        raise DuplicateModelsError("dup", {"models": ["a"]})

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/_boom")
    assert resp.status_code == 422
    assert resp.json() == {
        "error": {"code": "duplicate_models", "message": "dup", "details": {"models": ["a"]}}
    }
