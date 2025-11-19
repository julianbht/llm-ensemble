"""InferredDataset - set of samples actually processed by an infer run.

This represents the specific collection of judging samples that were inferred over
in a particular infer run. Multiple infer runs can produce the same InferredDataset
(same fingerprint) enabling idempotent re-runs.

Samples are always stored in deterministic order (sorted by sample.id) to support
reproducible slicing and consistent aggregation.
"""

from __future__ import annotations
from uuid import UUID
from pydantic import BaseModel, Field, field_validator

from llm_ensemble.ingest.schemas import JudgingSample
from llm_ensemble.libs.db import (
    compute_normalized_dataset_fingerprint,
    compute_normalized_dataset_uuid,
)


class InferredDataset(BaseModel):
    """Set of samples actually processed by an infer run.

    Represents the working set for inference - could be a subset, slice, or
    full set of samples from the source NormalizedDataset. The fingerprint
    enables:
    - Idempotent infer runs (same samples = same fingerprint = same entity)
    - Reproducible sample ordering for aggregation
    - Efficient validation in aggregate CLI (compare fingerprints)

    The fingerprint is computed from the sorted list of sample IDs (UUIDs).
    Samples are always stored sorted by sample.id to ensure reproducibility.

    Each InferredDataset references its source NormalizedDataset for provenance.
    """

    id: UUID = Field(
        ...,
        description="Deterministic UUID computed from fingerprint"
    )
    fingerprint: str = Field(
        ...,
        description="SHA256 hash of sorted sample IDs (deterministic identifier)"
    )
    normalized_dataset_id: UUID = Field(
        ...,
        description="Source NormalizedDataset this was derived from"
    )
    samples: list[JudgingSample] = Field(
        ...,
        description="Judging samples that were inferred, always sorted by sample.id"
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
        normalized_dataset_id: UUID,
    ) -> "InferredDataset":
        """Create InferredDataset with computed fingerprint and ID.

        Args:
            samples: List of judging samples (will be sorted by ID)
            normalized_dataset_id: Source NormalizedDataset UUID

        Returns:
            InferredDataset with computed fingerprint and deterministic ID
        """
        # Sort samples by ID for deterministic ordering
        sorted_samples = sorted(samples, key=lambda s: s.id)

        # Compute fingerprint from sorted sample IDs (reuse same function as NormalizedDataset)
        fingerprint = compute_normalized_dataset_fingerprint(sorted_samples)

        # Compute deterministic UUID from fingerprint (reuse same namespace)
        dataset_id = compute_normalized_dataset_uuid(fingerprint)

        return cls(
            id=dataset_id,
            fingerprint=fingerprint,
            normalized_dataset_id=normalized_dataset_id,
            samples=sorted_samples,
        )

    @property
    def sample_count(self) -> int:
        """Get number of samples in this dataset."""
        return len(self.samples)
