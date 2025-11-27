"""JudgedDataset - set of LLM judgements produced during inference.

This is the output of the infer pipeline and represents the actual judgements
from a single model configuration.

Provenance to input NormalizedDataset is tracked via:
  JudgedDataset → InferRun → IngestRun → NormalizedDataset

For resumability, the orchestrator can:
1. Load NormalizedDataset via InferRun.ingest_run_id
2. Compare its samples vs judgements' samples to find what's missing
"""

from __future__ import annotations
from uuid import UUID
from pydantic import BaseModel, Field

from llm_ensemble.infer.schemas.llm_judgement import LLMJudgement
from llm_ensemble.libs.db import compute_judged_dataset_fingerprint


class JudgedDataset(BaseModel):
    """Set of LLM judgements produced during inference with a single model config.

    The sample_fingerprint is computed from the sorted list of dataset_sample IDs
    (via llm_judgement → llm_prompt_text → dataset_sample).
    This identifies which samples were judged, independent of model/prompt used.

    Provenance to input NormalizedDataset is tracked via:
      JudgedDataset → InferRun → IngestRun → NormalizedDataset

    For resumability: Load NormalizedDataset via InferRun.ingest_run_id,
    then compare samples vs judgements.
    """

    id: UUID = Field(
        ...,
        description="Same as InferRun.id (1:1 relationship)"
    )
    model_config_id: UUID = Field(
        ...,
        description="Which model configuration was used for all judgements"
    )
    sample_fingerprint: str = Field(
        ...,
        description="SHA256 hash of sorted dataset_sample IDs (deterministic identifier)"
    )
    llm_judgements: list[LLMJudgement] = Field(
        ...,
        description="LLM judgements, one per dataset_sample"
    )

    @classmethod
    def create(
        cls,
        id: UUID,
        model_config_id: UUID,
        llm_judgements: list[LLMJudgement]
    ) -> "JudgedDataset":
        """Create JudgedDataset with computed sample_fingerprint.

        Args:
            id: JudgedDataset ID (same as InferRun.id)
            model_config_id: Which model config was used
            llm_judgements: List of LLM judgements

        Returns:
            JudgedDataset with computed sample_fingerprint

        Note: sample_fingerprint is computed from sorted dataset_sample IDs
        (via llm_judgement → llm_prompt_text → dataset_sample).
        """
        # Extract dataset_sample IDs for fingerprint computation
        dataset_sample_ids = [
            j.llm_prompt.dataset_sample.id
            for j in llm_judgements
        ]

        # Compute fingerprint from sorted dataset_sample IDs
        sample_fingerprint = compute_judged_dataset_fingerprint(dataset_sample_ids)

        return cls(
            id=id,
            model_config_id=model_config_id,
            sample_fingerprint=sample_fingerprint,
            llm_judgements=llm_judgements,
        )

    @property
    def judgement_count(self) -> int:
        """Get number of LLM judgements in this dataset."""
        return len(self.llm_judgements)
