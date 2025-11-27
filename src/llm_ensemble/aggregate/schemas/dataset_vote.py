"""DatasetVote - single position in an aggregated dataset.

Represents one query-document pair in the aggregated dataset, linking to its
aggregated results. Similar to DatasetSample in the ingest pipeline.
"""

from __future__ import annotations
from uuid import UUID
from pydantic import BaseModel, Field

from llm_ensemble.aggregate.schemas.aggregated_vote import AggregatedVote


class DatasetVote(BaseModel):
    """Single position in an aggregated dataset.

    Contains:
    - id: Deterministic UUID from (aggregated_dataset_id, sequence_number)
    - aggregated_dataset_id: Which aggregated dataset this belongs to
    - sequence_number: Position in the aggregated dataset (0-indexed)
    - aggregated_votes: Results from applying aggregation strategies

    This is similar to DatasetSample in the ingest pipeline, providing a
    stable position reference for query-document pairs in the aggregated dataset.
    """

    id: UUID = Field(
        ...,
        description="Deterministic UUID computed from aggregated_dataset_id and sequence_number"
    )

    aggregated_dataset_id: UUID = Field(
        ...,
        description="Which aggregated dataset this vote belongs to"
    )

    sequence_number: int = Field(
        ...,
        ge=0,
        description="Position in the aggregated dataset (0-indexed, preserves order)"
    )

    aggregated_votes: list[AggregatedVote] = Field(
        default_factory=list,
        description="Results from applying aggregation strategies to this query-document pair"
    )
