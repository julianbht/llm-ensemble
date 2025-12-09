"""NormalizedDataset - internal dataset with deterministic fingerprint.

This is the output of the DatasetReader and represents the "internal dataset" -
a specific collection of judging samples. Multiple ingest runs can produce the
same NormalizedDataset (same fingerprint) enabling idempotent re-runs.

Samples are always stored in deterministic order (sorted by sample.id) to support
reproducible slicing via future start_sample/end_sample parameters.

Design:
- All samples within one NormalizedDataset are from one external dataset
- external_dataset_name tracks the source dataset for context
"""

from __future__ import annotations
from uuid import UUID, uuid4
from typing import Optional
import hashlib
from pydantic import BaseModel, Field, field_validator

from llm_ensemble.ingest.schemas import JudgingSample
from llm_ensemble.ingest.schemas.dataset_sample import DatasetSample


class NormalizedDataset(BaseModel):
    """Internal dataset with fingerprint for deduplication.

    Represents a specific collection of dataset samples with a fingerprint
    computed from sorted judging sample IDs. This enables:
    - Deduplication detection via database unique constraint on fingerprint
    - Reproducible sample ordering for start/end slicing
    - Efficient validation in aggregate CLI (compare fingerprints)

    The fingerprint is computed from the sorted list of judging sample IDs (UUIDs).
    Samples are wrapped in DatasetSample objects that track position and dataset context,
    and are always stored sorted by judging_sample.id to ensure reproducibility.

    All samples within one NormalizedDataset are from one external dataset,
    tracked via external_dataset_name for context.
    """

    id: UUID = Field(
        default_factory=uuid4,
        description="Random UUID identifier"
    )
    fingerprint: str = Field(
        ...,
        description="SHA256 hash of sorted sample IDs for deduplication detection"
    )
    external_dataset_name: Optional[str] = Field(
        None,
        description="Name of the external source dataset (e.g., 'msmarco', 'llmjudge')"
    )
    samples: list[DatasetSample] = Field(
        ...,
        description="Dataset samples with position info, sorted by judging_sample.id for reproducibility"
    )

    @field_validator('samples')
    @classmethod
    def validate_samples_sorted(cls, v: list[DatasetSample]) -> list[DatasetSample]:
        """Ensure samples are sorted by judging_sample.id for deterministic ordering."""
        if not v:
            return v

        # Check if already sorted by judging_sample.id
        sample_ids = [s.judging_sample.id for s in v]
        if sample_ids != sorted(sample_ids):
            raise ValueError("Samples must be sorted by judging_sample.id for deterministic ordering")

        return v

    @classmethod
    def create(
        cls,
        samples: list[JudgingSample],
        external_dataset_name: Optional[str] = None
    ) -> "NormalizedDataset":
        """Create NormalizedDataset with computed fingerprint and random ID.

        Args:
            samples: List of judging samples (will be sorted by ID and wrapped in DatasetSample)
            external_dataset_name: Optional name of the external source dataset

        Returns:
            NormalizedDataset with computed fingerprint and random UUID
        """

        # Generate random UUID for this dataset
        dataset_id = uuid4()

        # Sort samples by ID for deterministic ordering
        sorted_samples = sorted(samples, key=lambda s: s.id)

        # Compute fingerprint from sorted sample IDs
        sample_ids_str = ":".join(str(s.id) for s in sorted_samples)
        fingerprint = hashlib.sha256(sample_ids_str.encode()).hexdigest()

        # Wrap each JudgingSample in a DatasetSample with sequence number
        dataset_samples = [
            DatasetSample(
                normalized_dataset_id=dataset_id,
                judging_sample=sample,
                sequence_number=idx,
            )
            for idx, sample in enumerate(sorted_samples)
        ]

        return cls(
            id=dataset_id,
            fingerprint=fingerprint,
            external_dataset_name=external_dataset_name,
            samples=dataset_samples,
        )

    @property
    def sample_count(self) -> int:
        """Get number of samples in this dataset."""
        return len(self.samples)
