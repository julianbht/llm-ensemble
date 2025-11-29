"""Port interface for aggregation strategy adapters.

Defines the abstract contract that all aggregation strategy adapters must implement.
This allows the system to use different aggregation methods (majority vote, weighted,
etc.) without coupling to specific implementations.

Uses template method pattern: aggregate() is concrete and handles DTO→Domain mapping,
subclasses implement aggregate_raw() with pure voting logic.

Adapter owns its identity via strategy_name property - no ID injection from config.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

from llm_ensemble.infer.schemas.llm_judgement import LLMJudgement
from llm_ensemble.aggregate.schemas.aggregated_vote import AggregatedVote
from llm_ensemble.aggregate.schemas.aggregation_strategy import AggregationStrategy as AggregationStrategyEntity
from llm_ensemble.libs.db import compute_aggregation_spec_uuid


class AggregationStrategyPort(ABC):
    """Abstract base class for aggregation strategy adapters with built-in mapping.

    Implementations provide voting logic in aggregate_raw(), which returns a simple dict.
    The base class handles conversion to AggregatedVote domain objects.

    Template Method Pattern:
    - aggregate() (concrete): calls aggregate_raw() and creates domain object
    - aggregate_raw() (abstract): subclasses implement voting logic
    - strategy_name (abstract): adapter defines its own identity

    This separates pure algorithm logic from domain object creation.
    Adapter owns its identity - no ID injection needed from config.
    """

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Natural key for AggregationStrategy entity.

        This is the canonical identifier for the strategy (e.g., 'majority_vote').
        Used to create AggregationStrategy domain entity.
        """
        pass

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

    @property
    @abstractmethod
    def name(self) -> str:
        """Return human-readable name of this strategy for logging."""
        pass
