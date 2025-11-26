"""JudgedDataset - set of LLM judgements produced during inference.

This is the output of the infer pipeline and represents the actual judgements
(LLMCalls with scores).

Provenance to input NormalizedDataset is tracked via:
  JudgedDataset → InferRun → IngestRun → NormalizedDataset

For resumability, the orchestrator can:
1. Load NormalizedDataset via InferRun.ingest_run_id
2. Compare its samples vs judgements' samples to find what's missing
"""

from __future__ import annotations
from uuid import UUID
from pydantic import BaseModel, Field, field_validator

from llm_ensemble.infer.schemas.llm_judgement import LLMJudgement
from llm_ensemble.libs.db import compute_judged_dataset_uuid
from llm_ensemble.libs.db import compute_judged_dataset_fingerprint


class JudgedDataset(BaseModel):
    """Set of LLM judgements produced during inference.

    The fingerprint is computed from the sorted list of dataset_sample IDs.
    This identifies which samples were judged, independent of model/prompt used.
    Judgements are stored sorted by their dataset_sample.id for reproducibility.

    Provenance to input NormalizedDataset is tracked via:
      JudgedDataset → InferRun → IngestRun → NormalizedDataset

    For resumability: Load NormalizedDataset via InferRun.ingest_run_id,
    then compare samples vs judgements.
    """

    id: UUID = Field(
        ...,
        description="Deterministic UUID computed from fingerprint"
    )
    fingerprint: str = Field(
        ...,
        description="SHA256 hash of sorted dataset_sample IDs (deterministic identifier)"
    )
    judgements: list[LLMJudgement] = Field(
        ...,
        description="LLM judgements, always sorted by dataset_sample.id for reproducibility"
    )

    @field_validator('judgements')
    @classmethod
    def validate_judgements_sorted(cls, v: list[LLMJudgement]) -> list[LLMJudgement]:
        """Ensure judgements are sorted by dataset_sample ID for deterministic ordering."""
        if not v:
            return v

        # Check if already sorted by dataset_sample.id
        # dataset_sample is nested in llm_prompt
        sample_ids = [j.llm_prompt.dataset_sample.id for j in v]
        if sample_ids != sorted(sample_ids):
            raise ValueError("Judgements must be sorted by dataset_sample.id for deterministic ordering")

        return v

    @classmethod
    def create(
        cls,
        judgements: list[LLMJudgement]
    ) -> "JudgedDataset":
        """Create JudgedDataset with computed fingerprint and ID.

        Args:
            judgements: List of LLM judgements (will be sorted by dataset_sample.id)

        Returns:
            JudgedDataset with computed fingerprint and deterministic ID

        Note: Fingerprint is computed from sorted dataset_sample IDs, which
        identifies which samples were judged (independent of model/prompt).
        """

        # Sort judgements by dataset_sample.id for deterministic ordering
        sorted_judgements = sorted(
            judgements,
            key=lambda j: j.llm_prompt.dataset_sample.id
        )

        # Extract dataset_sample IDs for fingerprint computation
        dataset_sample_ids = [
            j.llm_prompt.dataset_sample.id
            for j in sorted_judgements
        ]

        # Compute fingerprint from sorted dataset_sample IDs
        fingerprint = compute_judged_dataset_fingerprint(dataset_sample_ids)

        # Compute deterministic UUID from fingerprint
        dataset_id = compute_judged_dataset_uuid(fingerprint)

        return cls(
            id=dataset_id,
            fingerprint=fingerprint,
            judgements=sorted_judgements,
        )

    @property
    def sample_count(self) -> int:
        """Get number of judgements in this dataset."""
        return len(self.judgements)
