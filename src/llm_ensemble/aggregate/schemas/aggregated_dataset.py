"""AggregatedDataset - set of aggregated votes produced during aggregation.

This is the output of the aggregate pipeline and represents the aggregated results.
Similar to NormalizedDataset in ingest and InferRunOutput in infer.
"""

from __future__ import annotations
import hashlib
from uuid import UUID, uuid4
from pydantic import BaseModel, Field

from llm_ensemble.aggregate.schemas.aggregated_vote import AggregatedVote


class AggregatedDataset(BaseModel):
    """Set of aggregated votes produced during aggregation.

    The fingerprint is computed from the sorted list of dataset_sample IDs.
    This identifies which query-document pairs were aggregated, independent of
    which aggregation strategy was used.
    """

    id: UUID = Field(
        default_factory=uuid4,
        description="Random UUID identifier"
    )

    fingerprint: str = Field(
        default="",
        description="SHA256 hash of sorted dataset_sample IDs (computed automatically)"
    )

    aggregated_votes: list[AggregatedVote] = Field(
        default_factory=list,
        description="Aggregated votes (one per dataset_sample per aggregation_spec)"
    )

    @classmethod
    def create(
        cls,
        aggregated_votes: list[AggregatedVote]
    ) -> "AggregatedDataset":
        """Create AggregatedDataset with computed fingerprint.

        Args:
            aggregated_votes: List of aggregated votes

        Returns:
            AggregatedDataset with computed fingerprint and random UUID

        Note: Fingerprint is computed from sorted dataset_sample IDs, which
        identifies which query-document pairs were aggregated.
        """
        # Extract unique dataset_sample IDs from aggregated votes
        dataset_sample_ids = set()
        for vote in aggregated_votes:
            if vote.llm_judgements:
                # All judgements in a vote are for the same sample, so take first
                dataset_sample_id = vote.llm_judgements[0].llm_prompt.dataset_sample.id
                dataset_sample_ids.add(dataset_sample_id)

        # Sort for deterministic fingerprint
        sorted_sample_ids = sorted(dataset_sample_ids)

        # Compute fingerprint from sorted dataset_sample IDs
        sample_ids_str = ":".join(str(sid) for sid in sorted_sample_ids)
        fingerprint = hashlib.sha256(sample_ids_str.encode()).hexdigest()

        return cls(
            fingerprint=fingerprint,
            aggregated_votes=aggregated_votes,
        )

    @property
    def vote_count(self) -> int:
        """Get number of votes in this dataset."""
        return len(self.aggregated_votes)
