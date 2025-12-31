"""InferRunOutput - judgements and metrics produced during inference.

This is the output of the infer pipeline. It contains:
- The actual LLM judgements produced
- Sample fingerprint (which samples were judged)
- Aggregate metrics (counts, latencies, errors)

Configuration reference is via InferRun (not stored here).

This is separate from:
- InferRunInfo: Run metadata (git info, timestamps)
- InferRunConfig: Configuration used (stored in InferRun, not here)

For resumability, the orchestrator can:
1. Load NormalizedDataset via InferRunConfig.input_run_name
2. Compare its samples vs judgements' samples to find what's missing
"""

from __future__ import annotations
from uuid import UUID, uuid4
from typing import Optional
from pydantic import BaseModel, Field

from llm_ensemble.infer.domain.entities.llm_judgement import LLMJudgement


class InferRunOutput(BaseModel):
    """Output produced during an infer run.

    Contains:
    - Judgements: The actual LLM judgements produced
    - Sample fingerprint: Which samples were judged (deterministic identifier)
    - Aggregate metrics: Counts, latencies, error rates, warnings

    Configuration reference is via InferRun (not stored here).
    1:1 relationship with InferRun (same ID).
    """

    id: UUID = Field(
        default_factory=uuid4,
        description="Random UUID for this output (same as InferRun.id)"
    )

    llm_judgements: list[LLMJudgement] = Field(
        ...,
        description="LLM judgements, one per dataset_sample"
    )

    sample_fingerprint: Optional[str] = Field(
        default=None,
        description="SHA256 hash of sorted dataset_sample IDs (deterministic identifier, NULL until finalized)"
    )

    finished: bool = Field(
        default=False,
        description="Whether the run completed successfully (set to True in close())"
    )

    # Aggregate metrics (computed after run completes)
    judgement_count: int = Field(
        ...,
        description="Total number of judgements produced"
    )

    error_count: int = Field(
        default=0,
        description="Number of parsing failures or errors"
    )

    avg_latency_ms: float = Field(
        default=0.0,
        description="Average latency per judgement in milliseconds"
    )

    issues_summary: Optional[dict[str, int]] = Field(
        default=None,
        description="Summary of warnings collected during inference (warning_type -> count)"
    )
