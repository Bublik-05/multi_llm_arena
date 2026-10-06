"""Overall score, ranking and winner. Pure functions: no I/O, no LLM involved."""

from collections.abc import Mapping

from pydantic import BaseModel

from app.core.constants import CRITERIA_WEIGHTS
from app.schemas.judge import CriterionScores


class ScoredResponse(BaseModel):
    model: str
    scores: CriterionScores
    overall: float
    rank: int  # 1 = winner


def compute_overall(scores: CriterionScores) -> float:
    """Weighted sum using constants.CRITERIA_WEIGHTS.

    Rounded to 2 decimals: weights have 2 decimals, so this removes float noise and makes
    exact-tie comparisons reliable.
    """
    return round(sum(w * getattr(scores, name) for name, w in CRITERIA_WEIGHTS.items()), 2)


def rank_responses(
    scores_by_model: Mapping[str, CriterionScores], latency_ms_by_model: Mapping[str, float]
) -> list[ScoredResponse]:
    """Order by overall desc; ties: higher correctness, then lower latency, then name.

    The name is only there to make the order deterministic.
    """
    entries = [(model, s, compute_overall(s)) for model, s in scores_by_model.items()]
    entries.sort(key=lambda e: (-e[2], -e[1].correctness, latency_ms_by_model[e[0]], e[0]))
    return [
        ScoredResponse(model=model, scores=s, overall=overall, rank=position)
        for position, (model, s, overall) in enumerate(entries, start=1)
    ]
