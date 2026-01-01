"""InferRunSummary schema - extends base RunSummary with infer-specific metrics.

This contains inference-specific aggregate statistics computed after the run
completes, separate from InferRunInfo which contains immutable configuration
known before the run starts.
"""

from __future__ import annotations
from typing import Optional
from pydantic import Field

from llm_ensemble.libs.runtime.run_summary import RunSummary
from llm_ensemble.infer.application.write_summary import WriteSummary


class InferRunSummary(RunSummary):
    """Summary for infer CLI runs - aggregate metrics computed post-run.

    Extends the base RunSummary with inference-specific aggregate statistics:
    - Write summary (files written, records persisted)
    - Judgement counts (total, failed parses)
    - Latency statistics (total, average)
    - Cost statistics (total cost estimate in USD)
    - Token statistics (total prompt, completion, and combined tokens)
    - Warnings summary (counts by warning type)

    This is separate from InferRunInfo which contains immutable configuration.
    The InferRunInfo is persisted separately (infer_run_info.json) to avoid
    duplication. This summary contains only runtime metrics for quick inspection.
    """

    # Aggregate statistics (computed at end of run)
    write_summary: WriteSummary = Field(
        ...,
        description="Summary of write operations (judgements written to disk)"
    )

    judgement_count: int = Field(
        ...,
        description="Number of judgements produced"
    )

    failed_parses_count: int = Field(
        ...,
        description="Number of judgements where parsing failed (llm_score is None or label is None)"
    )

    total_latency_ms: float = Field(
        ...,
        description="Total latency across all judgements in milliseconds"
    )

    avg_latency_ms: float = Field(
        ...,
        description="Average latency per judgement in milliseconds"
    )

    total_cost_usd: Optional[float] = Field(
        None,
        ge=0.0,
        description="Total estimated cost in USD for all LLM inference calls"
    )

    total_prompt_tokens: Optional[int] = Field(
        None,
        ge=0,
        description="Total number of prompt tokens across all inference calls"
    )

    total_completion_tokens: Optional[int] = Field(
        None,
        ge=0,
        description="Total number of completion tokens across all inference calls"
    )

    total_tokens: Optional[int] = Field(
        None,
        ge=0,
        description="Total number of tokens (prompt + completion) across all inference calls"
    )

    issues_summary: Optional[dict[str, int]] = Field(
        default=None,
        description=(
            "Summary of warnings collected during inference run. "
            "Maps warning type name to count (e.g., {'ParserIssue': 45, 'ProviderWarning': 12}). "
            "Aggregated from all judgements at end of run."
        )
    )
