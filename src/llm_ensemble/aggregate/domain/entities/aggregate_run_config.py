"""AggregateRunConfig - immutable configuration bundle for an aggregate run.

Pure Pydantic model containing all configuration needed to execute aggregation:
- Aggregation strategy (which strategy to use, e.g., majority_vote)
- I/O configuration (which I/O adapter to use)
- Input runs (which infer runs to aggregate)

This is a serializable domain entity (frozen Pydantic model).
Separate from AggregateRun (git info, timestamps, run metadata).
"""

from __future__ import annotations
from uuid import UUID, uuid4
from pydantic import BaseModel, ConfigDict, Field


class AggregateRunConfig(BaseModel):
    """Immutable configuration bundle for an aggregate run.

    Pure Pydantic model - no business logic, just data.
    Contains configuration decisions made for this specific aggregation run.
    """

    id: UUID = Field(
        default_factory=uuid4,
        description="Random UUID for this config bundle"
    )

    aggregation_strategy_name: str = Field(
        ...,
        description="Name of the aggregation strategy used (e.g., 'majority_vote')"
    )

    io_config_name: str = Field(
        ...,
        description="Name of the I/O config used (e.g., 'json', 'db')"
    )

    input_run_names: list[str] = Field(
        ...,
        description="List of infer run identifiers to read judgements from"
    )

    model_config = ConfigDict(frozen=True)
