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

    The fingerprint is computed from the sorted list of LLMCall IDs (UUIDs).
    Judgements are stored sorted by their judging_sample.id for reproducibility.

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
        description="SHA256 hash of sorted LLMCall IDs (deterministic identifier)"
    )
    judgements: list[LLMJudgement] = Field(
        ...,
        description="LLM judgements, always sorted by judging_sample.id for reproducibility"
    )

    @field_validator('judgements')
    @classmethod
    def validate_judgements_sorted(cls, v: list[LLMJudgement]) -> list[LLMJudgement]:
        """Ensure judgements are sorted by LLMCall ID for deterministic ordering."""
        if not v:
            return v

        # Check if already sorted by llm_call_id
        # Note: LLMJudgement domain objects don't have llm_call_id directly,
        # but they will be created with deterministic IDs based on (request_id, infer_run_id)
        # For now, we'll sort by judging_sample.id to maintain deterministic ordering
        # This will be refined when we add llm_call_id to LLMJudgement domain object
        sample_ids = [j.judging_sample.id for j in v]
        if sample_ids != sorted(sample_ids):
            raise ValueError("Judgements must be sorted by judging_sample.id for deterministic ordering")

        return v

    @classmethod
    def create(
        cls,
        judgements: list[LLMJudgement]
    ) -> "JudgedDataset":
        """Create JudgedDataset with computed fingerprint and ID.

        Args:
            judgements: List of LLM judgements (will be sorted by sample ID)

        Returns:
            JudgedDataset with computed fingerprint and deterministic ID

        Note: Currently uses judging_sample.id for sorting since LLMJudgement
        domain objects don't expose llm_call_id. This maintains compatibility
        while we transition the structure. The fingerprint will be based on
        LLMCall IDs in the database layer.
        """

        # Sort judgements by sample ID for deterministic ordering
        # (LLMCall ID would be ideal, but not available in domain model yet)
        sorted_judgements = sorted(judgements, key=lambda j: j.judging_sample.id)

        # Create pseudo-objects with id attribute for fingerprint computation
        # In practice, this will be called with actual LLMCall ORMs or updated
        # domain objects that have llm_call_id
        class PseudoCall:
            def __init__(self, judgement_id):
                # For now, use judging_sample.id as proxy
                # This will be replaced with actual llm_call_id
                self.id = judgement_id

        pseudo_calls = [PseudoCall(j.judging_sample.id) for j in sorted_judgements]

        # Compute fingerprint from sorted call IDs
        fingerprint = compute_judged_dataset_fingerprint(pseudo_calls)

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
