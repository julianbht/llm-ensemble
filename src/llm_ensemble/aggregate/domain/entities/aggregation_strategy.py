"""AggregationStrategy entity - minimal entity tracking which strategy was used.

Domain entity (NOT adapter config).
Created from adapter's strategy_name property.
"""

from __future__ import annotations
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


class AggregationStrategy(BaseModel):
    """Minimal entity tracking which aggregation strategy was used.

    Just id + name - no wiring details (module/class paths).
    Name comes from adapter's strategy_name property (e.g., 'majority_vote').

    This is a domain entity, NOT an adapter config.
    """

    id: UUID = Field(
        default_factory=uuid4,
        description="Random UUID identifier"
    )

    name: str = Field(
        ...,
        description="Strategy name (e.g., 'majority_vote')"
    )
