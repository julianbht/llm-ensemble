"""JudgedDataset - set of LLM judgements produced during inference.

This is the output of the infer pipeline and represents the actual judgements
organized by position (DatasetJudgement).

Provenance to input NormalizedDataset is tracked via:
  JudgedDataset → InferRun → IngestRun → NormalizedDataset

For resumability, the orchestrator can:
1. Load NormalizedDataset via InferRun.ingest_run_id
2. Compare its samples vs judgements' samples to find what's missing
"""

from __future__ import annotations
from uuid import UUID
from pydantic import BaseModel, Field

from llm_ensemble.infer.schemas.dataset_judgement import DatasetJudgement
from llm_ensemble.libs.db import compute_judged_dataset_uuid
from llm_ensemble.libs.db import compute_judged_dataset_fingerprint


class JudgedDataset(BaseModel):
    """Set of dataset judgements produced during inference.

    The fingerprint is computed from the sorted list of dataset_judgement IDs.
    This identifies which samples were judged, independent of model/prompt used.
    Dataset judgements are stored sorted by sequence_number for reproducibility.

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
        description="SHA256 hash of sorted dataset_judgement IDs (deterministic identifier)"
    )
    dataset_judgements: list[DatasetJudgement] = Field(
        ...,
        description="Dataset judgements, sorted by sequence_number for reproducibility"
    )

    @classmethod
    def create(
        cls,
        dataset_judgements: list[DatasetJudgement]
    ) -> "JudgedDataset":
        """Create JudgedDataset with computed fingerprint and ID.

        Args:
            dataset_judgements: List of dataset judgements (will be sorted by sequence_number)

        Returns:
            JudgedDataset with computed fingerprint and deterministic ID

        Note: Fingerprint is computed from sorted dataset_judgement IDs.
        """
        # Sort by sequence_number for deterministic ordering
        sorted_judgements = sorted(
            dataset_judgements,
            key=lambda dj: dj.sequence_number
        )

        # Extract dataset_judgement IDs for fingerprint computation
        dataset_judgement_ids = [dj.id for dj in sorted_judgements]

        # Compute fingerprint from sorted dataset_judgement IDs
        fingerprint = compute_judged_dataset_fingerprint(dataset_judgement_ids)

        # Compute deterministic UUID from fingerprint
        dataset_id = compute_judged_dataset_uuid(fingerprint)

        return cls(
            id=dataset_id,
            fingerprint=fingerprint,
            dataset_judgements=sorted_judgements,
        )

    @property
    def judgement_count(self) -> int:
        """Get number of dataset judgements in this dataset."""
        return len(self.dataset_judgements)
