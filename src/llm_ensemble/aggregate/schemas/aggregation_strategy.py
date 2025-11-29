"""AggregationStrategy entity - minimal entity tracking which strategy was used.

Domain entity (NOT adapter config).
Created from adapter's strategy_name property.
"""

from __future__ import annotations
from uuid import UUID
from pydantic import BaseModel, Field

from llm_ensemble.libs.db import compute_aggregation_spec_uuid


class AggregationStrategy(BaseModel):
    """Minimal entity tracking which aggregation strategy was used.

    Just id + name - no wiring details (module/class paths).
    Name comes from adapter's strategy_name property (e.g., 'majority_vote').

    This is a domain entity, NOT an adapter config.
    """

    id: UUID = Field(
        ...,
        description="Deterministic UUID computed from strategy name"
    )

    name: str = Field(
        ...,
        description="Strategy name (e.g., 'majority_vote')"
    )

    @classmethod
    def create(cls, strategy_name: str) -> "AggregationStrategy":
        """Create AggregationStrategy entity with computed ID.

        Args:
            strategy_name: Natural key for the strategy (e.g., 'majority_vote')

        Returns:
            AggregationStrategy with computed ID
        """
        strategy_id = compute_aggregation_spec_uuid(strategy_name)
        return cls(id=strategy_id, name=strategy_name)
