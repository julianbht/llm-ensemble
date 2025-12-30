"""Domain logic for building AggregatedDataset with content-based fingerprinting.

Separates fingerprint computation (domain logic) from data structure (entity).
"""

from __future__ import annotations
import hashlib
from uuid import uuid4

from llm_ensemble.aggregate.domain.entities.aggregated_dataset import AggregatedDataset
from llm_ensemble.aggregate.domain.entities.aggregated_vote import AggregatedVote


def build_aggregated_dataset(
    aggregated_votes: list[AggregatedVote]
) -> AggregatedDataset:
    """Build AggregatedDataset with content-based fingerprint.

    Computes fingerprint from sorted dataset_sample IDs to enable deduplication
    across runs. The fingerprint identifies which query-document pairs were aggregated,
    independent of which aggregation strategy was used.

    Args:
        aggregated_votes: List of aggregated votes

    Returns:
        AggregatedDataset with content-based fingerprint and random UUID
    """
    # Generate random UUID for this dataset
    dataset_id = uuid4()

    # Extract unique dataset_sample IDs from aggregated votes
    dataset_sample_ids = set()
    for vote in aggregated_votes:
        if vote.llm_judgements:
            # All judgements in a vote are for the same sample, so take first
            dataset_sample_id = vote.llm_judgements[0].dataset_sample.id
            dataset_sample_ids.add(dataset_sample_id)

    # Sort for deterministic fingerprint
    sorted_sample_ids = sorted(dataset_sample_ids)

    # Compute fingerprint from sorted dataset_sample IDs
    sample_ids_str = ":".join(str(sid) for sid in sorted_sample_ids)
    fingerprint = hashlib.sha256(sample_ids_str.encode()).hexdigest()

    return AggregatedDataset(
        id=dataset_id,
        fingerprint=fingerprint,
        aggregated_votes=aggregated_votes,
    )
