"""Domain functions for mapping prompt-specific scores to standard RelevanceScore.

Contains pure domain logic for converting between different scoring systems
used by various prompts and the standardized RelevanceScore enumeration.
"""

from __future__ import annotations

from llm_ensemble.libs.schemas.relevance_score import RelevanceScore


def map_thomas_advanced_score(prompt_score: int) -> RelevanceScore:
    """Map thomas-advanced prompt score to standard RelevanceScore.

    The thomas-advanced prompt uses a 0-2 scale which maps to the
    standard 0-3 RelevanceScore enumeration as follows:

    Mapping:
    - 0 (not relevant) → IRRELEVANT (0)
    - 1 (relevant, partly helpful) → RELEVANT (1)
    - 2 (highly relevant, very helpful) → HIGHLY_RELEVANT (2)

    Args:
        prompt_score: Score from thomas-advanced prompt (0, 1, or 2)

    Returns:
        Mapped RelevanceScore enum value

    Raises:
        KeyError: If prompt_score is not in valid range (0-2)
    """
    mapping = {
        0: RelevanceScore.IRRELEVANT,
        1: RelevanceScore.RELEVANT,
        2: RelevanceScore.HIGHLY_RELEVANT,
    }
    return mapping[prompt_score]
