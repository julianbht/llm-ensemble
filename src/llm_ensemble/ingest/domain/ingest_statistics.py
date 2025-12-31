"""Domain functions for computing ingest run statistics.

Pure domain logic for calculating metrics from domain entities.
"""

from __future__ import annotations

from llm_ensemble.ingest.domain.entities.normalized_dataset import NormalizedDataset


def calculate_ingest_statistics(
    normalized_dataset: NormalizedDataset,
) -> tuple[int]:
    """Calculate ingest statistics from normalized dataset.

    Business rules:
    - Sample count = number of samples in normalized dataset

    Args:
        normalized_dataset: The normalized dataset entity produced

    Returns:
        Tuple of (sample_count,)
    """
    return (normalized_dataset.sample_count,)
