"""InferRunSummary schema - extends base RunSummary with infer-specific metrics.

This contains inference-specific aggregate statistics computed after the run
completes, separate from InferRunInfo which contains immutable configuration
known before the run starts.
"""

from __future__ import annotations
from typing import Optional
from pydantic import Field

from llm_ensemble.libs.runtime.run_summary import RunSummary
from llm_ensemble.infer.schemas.infer_run_info import InferRunInfo


class InferRunSummary(RunSummary):
    """Summary for infer CLI runs - aggregate metrics computed post-run.

    Extends the base RunSummary with inference-specific aggregate statistics:
    - Judgement counts (total, errors)
    - Latency statistics (total, average)
    - Warnings summary (counts by warning type)

    This is separate from InferRunInfo which contains immutable configuration.
    The InferRunInfo can be embedded in LLMJudgement objects immediately, while
    InferRunSummary is only computed and written at the end of the run.

    The summary includes the full InferRunInfo, so it contains both the runtime
    context (model config, prompt config, etc.) and the post-run metrics.
    """

    # Override to use InferRunInfo instead of base RunInfo
    run_info: InferRunInfo = Field(
        ...,
        description="Immutable inference run context (model config, prompt config, etc.)"
    )

    # Aggregate statistics (computed at end of run)
    judgement_count: int = Field(
        ...,
        description="Number of judgements produced"
    )

    error_count: int = Field(
        ...,
        description="Number of failed judgements (label=None)"
    )

    total_latency_ms: float = Field(
        ...,
        description="Total latency across all judgements in milliseconds"
    )

    avg_latency_ms: float = Field(
        ...,
        description="Average latency per judgement in milliseconds"
    )

    warnings_summary: Optional[dict[str, int]] = Field(
        default=None,
        description=(
            "Summary of warnings collected during inference run. "
            "Maps warning type name to count (e.g., {'ParserWarning': 45, 'ProviderWarning': 12}). "
            "Aggregated from all judgements at end of run."
        )
    )
