"""RelevanceScore enumeration for standardized relevance judgments."""

from __future__ import annotations
from enum import IntEnum


class RelevanceScore(IntEnum):
    """Standardized relevance score enumeration.

    Defines the four-level relevance scale used across all datasets.
    Integer values allow for numeric comparisons and aggregations.
    """

    IRRELEVANT = 0
    """Document is not relevant to the query."""

    RELEVANT = 1
    """Document has some relevance to the query."""

    HIGHLY_RELEVANT = 2
    """Document is highly relevant to the query."""

    PERFECTLY_RELEVANT = 3
    """Document is perfectly relevant to the query."""

    @property
    def label(self) -> str:
        """Get human-readable label for this relevance score.

        Returns:
            String label (e.g., "Perfectly Relevant")
        """
        labels = {
            0: "Irrelevant",
            1: "Relevant",
            2: "Highly Relevant",
            3: "Perfectly Relevant",
        }
        return labels[self.value]
