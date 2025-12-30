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
from pydantic import BaseModel, ConfigDict, Field, computed_field
from hashlib import sha256
import json


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

    @computed_field  # type: ignore[misc]
    @property
    def input_run_names_hash(self) -> str:
        """Compute SHA256 hash of sorted input_run_names for natural key.

        Sorting ensures deterministic hash regardless of input order.

        Returns:
            64-character hex digest (SHA256)
        """
        sorted_names = sorted(self.input_run_names)
        canonical = json.dumps(sorted_names, sort_keys=True)
        return sha256(canonical.encode()).hexdigest()

    model_config = ConfigDict(frozen=True)
