"""JudgedDataset - set of LLM judgements produced during inference.

This is the output of the infer pipeline and represents the actual judgements
from a single model configuration and adapter configuration.

Provenance to input NormalizedDataset is tracked via:
  JudgedDataset → InferRun → IngestRun → NormalizedDataset

For resumability, the orchestrator can:
1. Load NormalizedDataset via InferRun.ingest_run_id
2. Compare its samples vs judgements' samples to find what's missing
"""

from __future__ import annotations
import uuid
from uuid import UUID
from pydantic import BaseModel, Field

from llm_ensemble.infer.schemas.entities.llm_judgement import LLMJudgement
from llm_ensemble.infer.schemas.entities.adapter_config import AdapterConfig
from llm_ensemble.infer.schemas.model_config_schema import ModelConfig


class JudgedDataset(BaseModel):
    """Set of LLM judgements produced during inference.

    Captures both model configuration (which model) and adapter configuration
    (which prompt builder, parser, provider) used to produce all judgements.

    The sample_fingerprint is computed from the sorted list of dataset_sample IDs.
    This identifies which samples were judged, independent of model/adapters used.

    Provenance to input NormalizedDataset is tracked via:
      JudgedDataset → InferRun → IngestRun → NormalizedDataset

    For resumability: Load NormalizedDataset via InferRun.ingest_run_id,
    then compare samples vs judgements.
    """

    id: UUID = Field(
        default_factory=uuid.uuid4,
        description="Random UUID for this dataset"
    )

    model_config: ModelConfig = Field(
        ...,
        description="Model configuration used for all judgements in this dataset"
    )

    adapter_config: AdapterConfig = Field(
        ...,
        description="Adapter configuration (prompt builder, parser, provider) used for all judgements"
    )

    sample_fingerprint: str = Field(
        ...,
        description="SHA256 hash of sorted dataset_sample IDs (deterministic identifier)"
    )

    llm_judgements: list[LLMJudgement] = Field(
        ...,
        description="LLM judgements, one per dataset_sample"
    )

    @property
    def judgement_count(self) -> int:
        """Get number of LLM judgements in this dataset."""
        return len(self.llm_judgements)
