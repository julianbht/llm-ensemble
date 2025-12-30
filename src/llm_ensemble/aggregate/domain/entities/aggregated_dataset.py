"""AggregatedDataset - pure data entity for aggregation output.

This is the output of the aggregate pipeline and represents the aggregated results.
Similar to NormalizedDataset in ingest and InferRunOutput in infer.

Design:
- Pure data carrier - no business logic or factory methods
- Fingerprint computation handled by domain builder (aggregated_dataset_builder.py)
- Fingerprint computed from sorted dataset_sample IDs for deduplication
"""

from __future__ import annotations
from uuid import UUID, uuid4
from pydantic import BaseModel, Field

from llm_ensemble.aggregate.domain.entities.aggregated_vote import AggregatedVote


class AggregatedDataset(BaseModel):
    """Pure data entity representing aggregation output.

    Contains a collection of aggregated votes with a content-based fingerprint for
    deduplication. The fingerprint identifies which query-document pairs were aggregated,
    independent of which aggregation strategy was used.

    Use `build_aggregated_dataset()` from aggregated_dataset_builder.py to create instances.
    """

    id: UUID = Field(
        default_factory=uuid4,
        description="Random UUID identifier"
    )

    fingerprint: str = Field(
        ...,
        description="SHA256 hash of sorted dataset_sample IDs for deduplication"
    )

    aggregated_votes: list[AggregatedVote] = Field(
        ...,
        description="Aggregated votes (one per dataset_sample)"
    )

    @property
    def vote_count(self) -> int:
        """Get number of votes in this dataset."""
        return len(self.aggregated_votes)
