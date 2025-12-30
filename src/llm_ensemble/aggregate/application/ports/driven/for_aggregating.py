"""Port interface for aggregation strategy adapters.

Defines the abstract contract that all aggregation strategy adapters must implement.
Adapters translate aggregation logic concerns into AggregatedVote entities.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

from llm_ensemble.infer.domain.entities.llm_judgement import LLMJudgement
from llm_ensemble.aggregate.domain.entities.aggregated_vote import AggregatedVote
from llm_ensemble.aggregate.domain.entities.aggregation_strategy import AggregationStrategy


class ForAggregating(ABC):
    """Abstract interface for aggregation strategy adapters.

    Adapters implement this interface to build AggregatedVote domain entities
    from lists of LLM judgements. The adapter is responsible for:
    1. Applying the aggregation logic (internal implementation detail)
    2. Constructing and returning complete AggregatedVote entities
    3. Providing metadata about the strategy (via get_strategy())

    This follows proper hexagonal architecture - adapters (outer layer)
    depend on domain entities (inner layer), translating external concerns
    (aggregation algorithms) into domain concepts the service can work with.
    """

    @abstractmethod
    def aggregate(self, judgements: list[LLMJudgement]) -> AggregatedVote:
        """Apply aggregation strategy and create AggregatedVote domain entity.

        Applies the aggregation logic to the judgements and constructs
        an AggregatedVote domain entity with the results.

        Args:
            judgements: All model judgements for a single query-document pair

        Returns:
            AggregatedVote domain entity with consensus decision and metadata

        Note:
            Strategy should handle edge cases gracefully:
            - Empty judgements list
            - All models failed to parse (no valid labels)
            - Ties in voting
        """
        pass

    @abstractmethod
    def get_strategy(self) -> AggregationStrategy:
        """Get AggregationStrategy metadata for this adapter.

        Returns:
            AggregationStrategy entity with id and name
        """
        pass
