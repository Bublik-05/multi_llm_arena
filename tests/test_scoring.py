import pytest

from app.schemas.judge import CriterionScores
from app.services.evaluation.scoring import compute_overall, rank_responses


def cs(c: float, r: float, comp: float, cl: float) -> CriterionScores:
    return CriterionScores(correctness=c, relevance=r, completeness=comp, clarity=cl)


def test_overall_follows_documented_formula() -> None:
    # 0.40*8 + 0.25*6 + 0.20*7 + 0.15*10
    assert compute_overall(cs(8, 6, 7, 10)) == 7.6


@pytest.mark.parametrize(("value", "expected"), [(1, 1.0), (10, 10.0), (5, 5.0)])
def test_uniform_scores_give_the_same_overall(value: float, expected: float) -> None:
    assert compute_overall(cs(value, value, value, value)) == expected


def test_correctness_outweighs_clarity() -> None:
    assert compute_overall(cs(10, 1, 1, 1)) > compute_overall(cs(1, 1, 1, 10))


def test_ranking_orders_by_overall_and_numbers_from_one() -> None:
    ranked = rank_responses(
        {"low": cs(3, 3, 3, 3), "high": cs(9, 9, 9, 9), "mid": cs(6, 6, 6, 6)},
        {"low": 1, "high": 1, "mid": 1},
    )
    assert [r.model for r in ranked] == ["high", "mid", "low"]
    assert [r.rank for r in ranked] == [1, 2, 3]
    assert ranked[0].overall == 9.0


def test_exact_tie_is_broken_by_correctness() -> None:
    a, b = cs(10, 6, 5, 5), cs(5, 10, 10, 5)
    assert compute_overall(a) == compute_overall(b) == 7.25  # equal despite float arithmetic
    ranked = rank_responses({"b": b, "a": a}, {"a": 5000, "b": 1})
    assert [r.model for r in ranked] == ["a", "b"]  # correctness beats lower latency


def test_tie_on_overall_and_correctness_is_broken_by_latency() -> None:
    same = cs(8, 8, 8, 8)
    ranked = rank_responses({"slow": same, "fast": same}, {"slow": 3000, "fast": 1200})
    assert [r.model for r in ranked] == ["fast", "slow"]


def test_full_tie_is_deterministic() -> None:
    same = cs(8, 8, 8, 8)
    lat = {"x": 100, "y": 100}
    first = [r.model for r in rank_responses({"y": same, "x": same}, lat)]
    second = [r.model for r in rank_responses({"x": same, "y": same}, lat)]
    assert first == second == ["x", "y"]
