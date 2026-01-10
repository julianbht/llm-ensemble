"""InferRunSummary schema - extends base RunSummary with infer-specific metrics.

This contains inference-specific aggregate statistics computed after the run
completes, separate from InferRunInfo which contains immutable configuration
known before the run starts.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field

from llm_ensemble.libs.runtime.run_summary import RunSummary
from llm_ensemble.infer.application.write_summary import WriteSummary
from llm_ensemble.infer.domain.entities.infer_run import InferRun


class JudgementsSummary(BaseModel):
    """Summary of judgement counts and success rates."""

    total_count: int = Field(
        ...,
        ge=0,
        description="Total number of judgements produced"
    )

    failed_parses_count: int = Field(
        ...,
        ge=0,
        description="Number of judgements where parsing failed (llm_score is None or label is None)"
    )

    vote_breakdown: dict[str, int] = Field(
        ...,
        description=(
            "Count of judgements per relevance label "
            "(derived from RelevanceScore enum, excludes failed parses)"
        )
    )


class LatencySummary(BaseModel):
    """Summary of latency metrics."""

    total_ms: float = Field(
        ...,
        ge=0.0,
        description="Total latency across all judgements in milliseconds"
    )

    avg_ms: float = Field(
        ...,
        ge=0.0,
        description="Average latency per judgement in milliseconds"
    )


class CostSummary(BaseModel):
    """Summary of cost metrics."""

    total_usd: Optional[float] = Field(
        None,
        ge=0.0,
        description="Total estimated cost in USD for all LLM inference calls"
    )


class TokensSummary(BaseModel):
    """Summary of token usage metrics."""

    total_prompt: Optional[int] = Field(
        None,
        ge=0,
        description="Total number of prompt tokens across all inference calls"
    )

    total_completion: Optional[int] = Field(
        None,
        ge=0,
        description="Total number of completion tokens across all inference calls"
    )

    total: Optional[int] = Field(
        None,
        ge=0,
        description="Total number of tokens (prompt + completion) across all inference calls"
    )


class PerformanceSummary(BaseModel):
    """Summary of performance metrics (latency, cost, tokens)."""

    latency: LatencySummary = Field(
        ...,
        description="Latency statistics for all LLM calls"
    )

    cost: CostSummary = Field(
        ...,
        description="Cost statistics for all LLM calls"
    )

    tokens: TokensSummary = Field(
        ...,
        description="Token usage statistics for all LLM calls"
    )


class PersistenceSummary(BaseModel):
    """Summary of persistence operations with aggregated totals."""

    total_created: int = Field(
        ...,
        ge=0,
        description="Total number of entities created across all types"
    )

    total_skipped: int = Field(
        ...,
        ge=0,
        description="Total number of entities skipped (already existed) across all types"
    )

    details: WriteSummary = Field(
        ...,
        description="Detailed breakdown of write operations by entity type"
    )


class InferRunSummary(RunSummary):
    """Summary for infer CLI runs - aggregate metrics computed post-run.

    Extends the base RunSummary with inference-specific aggregate statistics:
    - Run: complete run entity (includes config, git info, timestamps, run metadata)
    - Judgements: counts and success metrics
    - Performance: latency, cost, and token usage
    - Persistence: write operations summary
    - Issues: warnings and errors collected during run
    """

    run: InferRun = Field(
        ...,
        description="Complete inference run entity (includes config, git info, timestamps, run metadata)"
    )

    judgements: JudgementsSummary = Field(
        ...,
        description="Summary of judgement counts and parsing success"
    )

    performance: PerformanceSummary = Field(
        ...,
        description="Performance metrics including latency, cost, and token usage"
    )

    persistence: PersistenceSummary = Field(
        ...,
        description="Summary of write operations to storage"
    )

    issues: Optional[dict[str, int]] = Field(
        default=None,
        description=(
            "Summary of warnings collected during inference run. "
            "Maps warning type name to count (e.g., {'ParserIssue': 45, 'ProviderWarning': 12}). "
            "Aggregated from all judgements at end of run."
        )
    )
