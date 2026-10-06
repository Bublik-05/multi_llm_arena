from app.core.config import Settings
from app.core.constants import CRITERIA_WEIGHTS


def test_sqlite_url_is_converted_to_async_driver() -> None:
    s = Settings(database_url="sqlite:///./x.db")
    assert s.async_database_url == "sqlite+aiosqlite:///./x.db"


def test_base_url_trailing_slash_stripped() -> None:
    assert Settings(ollama_base_url="http://h:11434/").ollama_base_url == "http://h:11434"


def test_env_override(monkeypatch) -> None:
    monkeypatch.setenv("MAX_MODELS", "4")
    assert Settings().max_models == 4


def test_weights_sum_to_one() -> None:
    assert abs(sum(CRITERIA_WEIGHTS.values()) - 1.0) < 1e-9
