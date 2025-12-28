"""Domain utilities for working with datasets during inference.

Pure domain logic for dataset operations - no adapter dependencies.
"""

from __future__ import annotations
from typing import Optional

from llm_ensemble.ingest.domain.entities.normalized_dataset import NormalizedDataset


def resolve_slice_indices(
    dataset: NormalizedDataset,
    start_idx: Optional[int] = None,
    end_idx: Optional[int] = None,
) -> tuple[int, int]:
    """Resolve optional slice indices to actual dataset boundaries.

    Converts None values to actual indices based on dataset size.
    Single source of truth for index resolution logic.

    Args:
        dataset: NormalizedDataset to resolve indices for
        start_idx: Start index (inclusive), None means from beginning
        end_idx: End index (exclusive), None means until end

    Returns:
        Tuple of (actual_start_idx, actual_end_idx)

    Examples:
        >>> resolve_slice_indices(dataset, None, None)  # (0, len(dataset.samples))
        >>> resolve_slice_indices(dataset, 0, 10)       # (0, 10)
        >>> resolve_slice_indices(dataset, 10, None)    # (10, len(dataset.samples))
    """
    actual_start_idx = start_idx if start_idx is not None else 0
    actual_end_idx = end_idx if end_idx is not None else len(dataset.samples)
    return actual_start_idx, actual_end_idx
