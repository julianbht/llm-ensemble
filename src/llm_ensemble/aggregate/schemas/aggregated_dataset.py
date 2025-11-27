"""AggregatedDataset - set of aggregated votes produced during aggregation.

This is the output of the aggregate pipeline and represents the aggregated results.
Similar to NormalizedDataset in ingest and JudgedDataset in infer.

Idempotent design: Multiple aggregate runs can produce the same AggregatedDataset
if they aggregate the same set of votes (identified by fingerprint).
"""

from __future__ import annotations
from uuid import UUID
from pydantic import BaseModel, Field

from llm_ensemble.aggregate.schemas.dataset_vote import DatasetVote
from llm_ensemble.libs.db import compute_aggregated_dataset_uuid, compute_aggregated_dataset_fingerprint


class AggregatedDataset(BaseModel):
    """Set of aggregated votes produced during aggregation.

    The fingerprint is computed from the sorted list of dataset_vote IDs.
    This identifies which query-document pairs were aggregated, independent of
    which aggregation strategy was used.

    Dataset_votes are stored sorted by their sequence_number for reproducibility.

    Idempotent: Multiple aggregate runs aggregating the same votes will produce
    the same AggregatedDataset (same fingerprint → same UUID).
    """

    id: UUID = Field(
        ...,
        description="Deterministic UUID computed from fingerprint"
    )

    fingerprint: str = Field(
        ...,
        description="SHA256 hash of sorted dataset_vote IDs (deterministic identifier)"
    )

    dataset_votes: list[DatasetVote] = Field(
        default_factory=list,
        description="Aggregated votes, sorted by sequence_number for reproducibility"
    )

    @classmethod
    def create(
        cls,
        dataset_votes: list[DatasetVote]
    ) -> "AggregatedDataset":
        """Create AggregatedDataset with computed fingerprint and ID.

        Args:
            dataset_votes: List of dataset votes (will be sorted by sequence_number)

        Returns:
            AggregatedDataset with computed fingerprint and deterministic ID

        Note: Fingerprint is computed from sorted dataset_vote IDs, which
        identifies which query-document pairs were aggregated.
        """
        # Sort by sequence_number for deterministic ordering
        sorted_votes = sorted(dataset_votes, key=lambda v: v.sequence_number)

        # Extract dataset_vote IDs for fingerprint computation
        dataset_vote_ids = [v.id for v in sorted_votes]

        # Compute fingerprint from sorted dataset_vote IDs
        fingerprint = compute_aggregated_dataset_fingerprint(dataset_vote_ids)

        # Compute deterministic UUID from fingerprint
        dataset_id = compute_aggregated_dataset_uuid(fingerprint)

        return cls(
            id=dataset_id,
            fingerprint=fingerprint,
            dataset_votes=sorted_votes,
        )

    @property
    def vote_count(self) -> int:
        """Get number of votes in this dataset."""
        return len(self.dataset_votes)
