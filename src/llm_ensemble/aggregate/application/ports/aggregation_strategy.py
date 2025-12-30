"""Port interface for aggregation strategy adapters.

Defines the abstract contract that all aggregation strategy adapters must implement.
This allows the system to use different aggregation methods (majority vote, weighted,
etc.) without coupling to specific implementations.

Uses template method pattern: aggregate() is concrete and handles DTO→Domain mapping,
subclasses implement aggregate_raw() with pure voting logic.

Strategy identity comes from config, not from adapter.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

from llm_ensemble.infer.domain.entities.llm_judgement import LLMJudgement
from llm_ensemble.aggregate.domain.entities.aggregated_vote import AggregatedVote
from llm_ensemble.aggregate.domain.entities.aggregation_strategy import AggregationStrategy as AggregationStrategyEntity


class AggregationStrategyPort(ABC):
    """Abstract base class for aggregation strategy adapters with built-in mapping.

    Implementations provide voting logic in aggregate_raw(), which returns a simple dict.
    The base class handles conversion to AggregatedVote domain objects.

    Template Method Pattern:
    - aggregate() (concrete): calls aggregate_raw() and creates domain object
    - aggregate_raw() (abstract): subclasses implement voting logic

    This separates pure algorithm logic from domain object creation.
    Strategy identity (strategy_name) comes from config and is passed to constructor.
    """

    def __init__(self, strategy_name: str):
        """Initialize strategy adapter with identity from config.

        Args:
            strategy_name: Natural key for AggregationStrategy entity (from config)
        """
        self.strategy_name = strategy_name

    @abstractmethod
    def aggregate_raw(
        self,
        judgements: list[LLMJudgement]
    ) -> dict:
        """Implement voting logic - return raw vote data.

        Subclasses implement this with pure voting logic.

        Args:
            judgements: All model judgements for a single query-document pair

        Returns:
            dict with keys: final_label, final_confidence, final_reasoning

        Note:
            Strategy should handle edge cases gracefully:
            - Empty judgements list
            - All models failed to parse (no valid labels)
            - Ties in voting
        """
        pass

    def aggregate(
        self,
        judgements: list[LLMJudgement]
    ) -> AggregatedVote:
        """Apply aggregation strategy to combine multiple model judgements.

        Public interface called by service. Internally calls aggregate_raw()
        and maps result to domain object.

        Args:
            judgements: All model judgements for a single query-document pair

        Returns:
            AggregatedVote with consensus decision and metadata
        """
        # Call subclass implementation (returns dict)
        vote_data = self.aggregate_raw(judgements)

        # Create AggregationStrategy entity from adapter's strategy_name
        aggregation_strategy = AggregationStrategyEntity.create(self.strategy_name)

        # Map to domain entity (port layer's responsibility)
        return AggregatedVote.create(
            aggregation_strategy=aggregation_strategy,
            llm_judgements=judgements,
            final_label=vote_data.get("final_label"),
            final_confidence=vote_data.get("final_confidence"),
            final_reasoning=vote_data.get("final_reasoning"),
        )
