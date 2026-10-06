"""Application settings loaded from environment variables / .env."""

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    default_judge_model: str = "qwen3"
    request_timeout: float = Field(default=120.0, gt=0)
    connect_timeout: float = Field(default=5.0, gt=0)

    # Database
    database_url: str = "sqlite:///./arena.db"

    # Limits
    max_models: int = Field(default=6, ge=2)
    max_prompt_chars: int = Field(default=8000, ge=1)
    max_concurrency: int = Field(default=4, ge=1)
    # Cap on tokens a participant may generate (also bounds latency of verbose models).
    participant_max_tokens: int = Field(default=2048, ge=1)
    # Judge context window and output cap. Ollama silently truncates prompts that exceed
    # num_ctx, so the judge service fits responses into this budget explicitly.
    judge_num_ctx: int = Field(default=8192, ge=1024)
    judge_max_tokens: int = Field(default=1024, ge=128)

    # App
    log_level: str = "INFO"
    cors_origins: list[str] = ["http://localhost:5173"]

    @field_validator("ollama_base_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    @field_validator("log_level")
    @classmethod
    def _upper_level(cls, v: str) -> str:
        return v.upper()

    @property
    def async_database_url(self) -> str:
        """Convert a plain sqlite URL to the aiosqlite driver used by the app."""
        if self.database_url.startswith("sqlite:///"):
            return self.database_url.replace("sqlite:///", "sqlite+aiosqlite:///", 1)
        return self.database_url


@lru_cache
def get_settings() -> Settings:
    return Settings()
