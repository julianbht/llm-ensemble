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
from uuid import UUID
from typing import Optional
from pydantic import BaseModel, Field, field_validator

from llm_ensemble.ingest.schemas import JudgingSample
from llm_ensemble.libs.db import compute_normalized_dataset_uuid
from llm_ensemble.libs.db import compute_normalized_dataset_fingerprint


class NormalizedDataset(BaseModel):
    """Internal dataset with deterministic fingerprint.

    Represents a specific collection of judging samples with a deterministic
    fingerprint computed from sorted sample IDs. This enables:
    - Idempotent ingest runs (same samples = same fingerprint = same entity)
    - Reproducible sample ordering for start/end slicing
    - Efficient validation in aggregate CLI (compare fingerprints)

    The fingerprint is computed from the sorted list of sample IDs (UUIDs).
    Samples are always stored sorted by sample.id to ensure reproducibility.

    All samples within one NormalizedDataset are from one external dataset,
    tracked via external_dataset_name for context.
    """

    id: UUID = Field(
        ...,
        description="Deterministic UUID computed from fingerprint"
    )
    fingerprint: str = Field(
        ...,
        description="SHA256 hash of sorted sample IDs (deterministic identifier)"
    )
    external_dataset_name: Optional[str] = Field(
        None,
        description="Name of the external source dataset (e.g., 'msmarco', 'llmjudge')"
    )
    samples: list[JudgingSample] = Field(
        ...,
        description="Judging samples, always sorted by sample.id for reproducibility"
    )

    @field_validator('samples')
    @classmethod
    def validate_samples_sorted(cls, v: list[JudgingSample]) -> list[JudgingSample]:
        """Ensure samples are sorted by ID for deterministic ordering."""
        if not v:
            return v

        # Check if already sorted
        sample_ids = [s.id for s in v]
        if sample_ids != sorted(sample_ids):
            raise ValueError("Samples must be sorted by sample.id for deterministic ordering")

        return v

    @classmethod
    def create(
        cls,
        samples: list[JudgingSample],
        external_dataset_name: Optional[str] = None
    ) -> "NormalizedDataset":
        """Create NormalizedDataset with computed fingerprint and ID.

        Args:
            samples: List of judging samples (will be sorted by ID)
            external_dataset_name: Optional name of the external source dataset

        Returns:
            NormalizedDataset with computed fingerprint and deterministic ID
        """

        # Sort samples by ID for deterministic ordering
        sorted_samples = sorted(samples, key=lambda s: s.id)

        # Compute fingerprint from sorted sample IDs
        fingerprint = compute_normalized_dataset_fingerprint(sorted_samples)

        # Compute deterministic UUID from fingerprint
        dataset_id = compute_normalized_dataset_uuid(fingerprint)

        return cls(
            id=dataset_id,
            fingerprint=fingerprint,
            external_dataset_name=external_dataset_name,
            samples=sorted_samples,
        )

    @property
    def sample_count(self) -> int:
        """Get number of samples in this dataset."""
        return len(self.samples)
