"""Domain constants. Scoring and rating parameters live here, not inside LLM prompts."""

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

# Evaluation criteria and their weights in the overall score.
CRITERIA_WEIGHTS: Final[Mapping[str, float]] = MappingProxyType(
    {
        "correctness": 0.40,
        "relevance": 0.25,
        "completeness": 0.20,
        "clarity": 0.15,
    }
)
if not math.isclose(sum(CRITERIA_WEIGHTS.values()), 1.0):
    raise RuntimeError("CRITERIA_WEIGHTS must sum to 1.0")

SCORE_MIN: Final[int] = 1
SCORE_MAX: Final[int] = 10

# Two responses whose overall scores differ by less than this are a draw.
DRAW_EPSILON: Final[float] = 0.05

# Elo
INITIAL_RATING: Final[float] = 1000.0
ELO_K_FACTOR: Final[float] = 32.0

# Minimum number of valid responses required to call the judge.
MIN_VALID_RESPONSES: Final[int] = 2

# Judge
JUDGE_MAX_ATTEMPTS: Final[int] = 3  # first try + 2 retries with validation feedback
MAX_REASONING_CHARS: Final[int] = 2000

# Context budgeting (rough, deliberately conservative: non-English text tokenizes worse).
CHARS_PER_TOKEN: Final[int] = 3
CONTEXT_SAFETY_TOKENS: Final[int] = 100
FEEDBACK_RESERVE_CHARS: Final[int] = 300  # room for the retry feedback line
MIN_RESPONSE_CHARS: Final[int] = 200  # never truncate a response below this
