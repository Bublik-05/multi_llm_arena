import pytest
from pydantic import ValidationError

from app.core.constants import CRITERIA_WEIGHTS
from app.schemas.judge import CriterionScores, JudgeOutput, build_judge_schema
from app.services.evaluation.prompts import CRITERION_DESCRIPTIONS

GOOD = {"correctness": 8, "relevance": 7, "completeness": 6, "clarity": 9}


def test_criteria_are_consistent_across_schema_weights_and_prompt() -> None:
    assert set(CriterionScores.model_fields) == set(CRITERIA_WEIGHTS)
    assert set(CRITERION_DESCRIPTIONS) == set(CRITERIA_WEIGHTS)


@pytest.mark.parametrize("value", [1, 10, 5.5])
def test_scores_accept_range(value: float) -> None:
    assert CriterionScores(**{**GOOD, "correctness": value}).correctness == value


@pytest.mark.parametrize("value", [0, 0.9, 10.1, 11, -3])
def test_scores_reject_out_of_range(value: float) -> None:
    with pytest.raises(ValidationError):
        CriterionScores(**{**GOOD, "correctness": value})


def test_scores_require_every_criterion() -> None:
    with pytest.raises(ValidationError):
        CriterionScores(correctness=5, relevance=5, completeness=5)  # no clarity


def test_scores_reject_non_numeric() -> None:
    with pytest.raises(ValidationError):
        CriterionScores(**{**GOOD, "clarity": "high"})


def test_judge_output_ignores_self_declared_winner_and_overall() -> None:
    out = JudgeOutput.model_validate(
        {"winner": "A", "reasoning": "r", "scores": {"A": {**GOOD, "overall": 9.9}}}
    )
    assert "winner" not in out.model_dump()
    assert "overall" not in out.scores["A"].model_dump()


def test_judge_output_null_reasoning_becomes_empty() -> None:
    assert JudgeOutput.model_validate({"reasoning": None, "scores": {}}).reasoning == ""


def test_judge_output_requires_scores() -> None:
    with pytest.raises(ValidationError):
        JudgeOutput.model_validate({"reasoning": "x"})


def test_generated_json_schema_is_strict_and_reasoning_comes_first() -> None:
    schema = build_judge_schema(["A", "B"])
    assert list(schema["properties"]) == ["reasoning", "scores"]
    scores = schema["properties"]["scores"]
    assert scores["required"] == ["A", "B"] and scores["additionalProperties"] is False
    per_response = scores["properties"]["A"]
    assert per_response["required"] == list(CRITERIA_WEIGHTS)
    assert per_response["properties"]["clarity"] == {
        "type": "integer",
        "minimum": 1,
        "maximum": 10,
    }
