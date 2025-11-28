"""Port interface for aggregation strategies.

Defines the abstract contract that all aggregation strategy adapters must implement.
This allows the system to use different aggregation methods (majority vote, weighted,
etc.) without coupling to specific implementations.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

from llm_ensemble.infer.schemas.llm_judgement import LLMJudgement
from llm_ensemble.aggregate.schemas.aggregated_vote import AggregatedVote


class AggregationStrategy(ABC):
    """Abstract base class for aggregation strategies.

    Implementations can use different aggregation methods (majority vote,
    weighted vote, soft voting, etc.) while providing a consistent interface
    to combine multiple model judgements into a consensus decision.

    Strategy adapters receive aggregation_spec_id in their constructor,
    similar to how infer adapters receive config context.
    """

    @abstractmethod
    def aggregate(
        self,
        judgements: list[LLMJudgement]
    ) -> AggregatedVote:
        """Apply aggregation strategy to combine multiple model judgements.

        Args:
            judgements: All model judgements for a single query-document pair

        Returns:
            AggregatedVote with consensus decision and metadata

        Note:
            Strategy should handle edge cases gracefully:
            - Empty judgements list
            - All models failed to parse (no valid labels)
            - Ties in voting
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return human-readable name of this strategy for logging."""
        pass
