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

    Computes fingerprint from sorted aggregated vote IDs to enable deduplication
    across runs. The fingerprint identifies which specific votes were produced,
    ensuring different aggregation runs with different judgements produce distinct datasets.

    Args:
        aggregated_votes: List of aggregated votes

    Returns:
        AggregatedDataset with content-based fingerprint and random UUID
    """
    # Generate random UUID for this dataset
    dataset_id = uuid4()

    # Sort vote IDs for deterministic fingerprint
    sorted_vote_ids = sorted(str(vote.id) for vote in aggregated_votes)

    # Compute fingerprint from sorted aggregated vote IDs
    vote_ids_str = ":".join(sorted_vote_ids)
    fingerprint = hashlib.sha256(vote_ids_str.encode()).hexdigest()

    return AggregatedDataset(
        id=dataset_id,
        fingerprint=fingerprint,
        aggregated_votes=aggregated_votes,
    )
