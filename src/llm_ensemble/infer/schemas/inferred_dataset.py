"""InferredDataset - set of samples processed during inference with judgements.

This is the output of the infer pipeline and represents the working set for
inference. Multiple infer runs can produce the same InferredDataset (same
fingerprint) enabling idempotent re-runs.

The fingerprint is computed from sorted sample IDs and enables aggregate CLI
to validate that all input runs processed the same samples.
"""

from __future__ import annotations
from uuid import UUID
from pydantic import BaseModel, Field, field_validator

from llm_ensemble.infer.schemas.llm_judgement import LLMJudgement
from llm_ensemble.libs.db import compute_normalized_dataset_uuid
from llm_ensemble.libs.db import compute_normalized_dataset_fingerprint


class InferredDataset(BaseModel):
    """Set of samples processed during inference with their judgements.

    Represents the working set for an inference run with a deterministic
    fingerprint computed from sorted sample IDs. This enables:
    - Idempotency: same samples = same fingerprint = same entity
    - Validation: aggregate CLI can verify all runs processed same samples
    - Reproducible ordering for deterministic processing

    The fingerprint is computed from the sorted list of sample IDs (UUIDs).
    Judgements are stored sorted by their judging_sample.id for reproducibility.
    """

    id: UUID = Field(
        ...,
        description="Deterministic UUID computed from fingerprint"
    )
    fingerprint: str = Field(
        ...,
        description="SHA256 hash of sorted sample IDs (deterministic identifier)"
    )
    judgements: list[LLMJudgement] = Field(
        ...,
        description="LLM judgements, always sorted by judging_sample.id for reproducibility"
    )

    @field_validator('judgements')
    @classmethod
    def validate_judgements_sorted(cls, v: list[LLMJudgement]) -> list[LLMJudgement]:
        """Ensure judgements are sorted by sample ID for deterministic ordering."""
        if not v:
            return v

        # Check if already sorted by judging_sample.id
        sample_ids = [j.judging_sample.id for j in v]
        if sample_ids != sorted(sample_ids):
            raise ValueError("Judgements must be sorted by judging_sample.id for deterministic ordering")

        return v

    @classmethod
    def create(cls, judgements: list[LLMJudgement]) -> "InferredDataset":
        """Create InferredDataset with computed fingerprint and ID.

        Args:
            judgements: List of LLM judgements (will be sorted by sample ID)

        Returns:
            InferredDataset with computed fingerprint and deterministic ID
        """

        # Sort judgements by sample ID for deterministic ordering
        sorted_judgements = sorted(judgements, key=lambda j: j.judging_sample.id)

        # Extract judging samples for fingerprint computation
        # (reuse same fingerprint computation as NormalizedDataset)
        samples = [j.judging_sample for j in sorted_judgements]

        # Compute fingerprint from sorted sample IDs
        fingerprint = compute_normalized_dataset_fingerprint(samples)

        # Compute deterministic UUID from fingerprint
        dataset_id = compute_normalized_dataset_uuid(fingerprint)

        return cls(
            id=dataset_id,
            fingerprint=fingerprint,
            judgements=sorted_judgements,
        )

    @property
    def sample_count(self) -> int:
        """Get number of samples in this dataset."""
        return len(self.judgements)
