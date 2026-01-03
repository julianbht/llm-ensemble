"""Domain logic for building NormalizedDataset with content-based fingerprinting.

Separates fingerprint computation (domain logic) from data structure (entity).
"""

from __future__ import annotations
import hashlib
from uuid import uuid4
from typing import Optional

from llm_ensemble.ingest.domain.entities.judging_sample import JudgingSample
from llm_ensemble.ingest.domain.entities.dataset_sample import NormalizedDatasetJudgingSample
from llm_ensemble.ingest.domain.entities.normalized_dataset import NormalizedDataset


def build_normalized_dataset(
    samples: list[JudgingSample],
    external_dataset_name: Optional[str] = None,
) -> NormalizedDataset:
    """Build NormalizedDataset with content-based fingerprint.

    Computes fingerprint from sample content (query/document hashes + scores)
    rather than random UUIDs, enabling proper deduplication across runs.

    Args:
        samples: List of judging samples (will be sorted by content for deterministic ordering)
        external_dataset_name: Optional name of the external source dataset

    Returns:
        NormalizedDataset with content-based fingerprint and wrapped samples
    """
    # Generate random UUID for this dataset (needed for DatasetSample references)
    dataset_id = uuid4()

    # Sort samples by content fingerprint for deterministic ordering
    # This ensures same samples in different order produce same fingerprint
    sorted_samples = sorted(
        samples,
        key=lambda s: f"{s.query.content_hash}:{s.document.content_hash}:{s.gold_score.value}"
    )

    # Compute fingerprint from sample content (NOT random IDs)
    # This enables deduplication: same data = same fingerprint
    sample_fingerprints = [
        f"{s.query.content_hash}:{s.document.content_hash}:{s.gold_score.value}"
        for s in sorted_samples
    ]
    fingerprint_input = ":".join(sample_fingerprints)
    fingerprint = hashlib.sha256(fingerprint_input.encode()).hexdigest()

    # Wrap each JudgingSample in a DatasetSample with sequence number
    dataset_samples = [
        NormalizedDatasetJudgingSample(
            normalized_dataset_id=dataset_id,
            judging_sample=sample,
            sequence_number=idx,
        )
        for idx, sample in enumerate(sorted_samples)
    ]

    return NormalizedDataset(
        id=dataset_id,
        fingerprint=fingerprint,
        external_dataset_name=external_dataset_name,
        samples=dataset_samples,
    )
