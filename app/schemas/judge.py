"""Schemas for what the judge returns. The judge only scores; it never picks a winner."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.constants import CRITERIA_WEIGHTS, SCORE_MAX, SCORE_MIN


def _score() -> Any:
    return Field(ge=SCORE_MIN, le=SCORE_MAX)


class CriterionScores(BaseModel):
    """Scores 1-10 per criterion. Field names must match constants.CRITERIA_WEIGHTS."""

    model_config = ConfigDict(extra="ignore")  # a judge that adds "overall" is not an error

    correctness: float = _score()
    relevance: float = _score()
    completeness: float = _score()
    clarity: float = _score()


class JudgeOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")  # ignore a self-declared "winner" etc.

    # Declared first on purpose: the model writes its comparison before committing to numbers.
    reasoning: str = ""
    scores: dict[str, CriterionScores]  # keyed by anonymous label

    @field_validator("reasoning", mode="before")
    @classmethod
    def _none_to_empty(cls, v: Any) -> Any:
        return "" if v is None else v


def build_judge_schema(labels: list[str]) -> dict[str, Any]:
    """JSON Schema passed to the backend to constrain the judge's output."""
    per_response: dict[str, Any] = {
        "type": "object",
        "properties": {
            c: {"type": "integer", "minimum": SCORE_MIN, "maximum": SCORE_MAX}
            for c in CRITERIA_WEIGHTS
        },
        "required": list(CRITERIA_WEIGHTS),
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "scores": {
                "type": "object",
                "properties": {label: per_response for label in labels},
                "required": list(labels),
                "additionalProperties": False,
            },
        },
        "required": ["reasoning", "scores"],
        "additionalProperties": False,
    }
