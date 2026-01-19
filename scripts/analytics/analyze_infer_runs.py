#!/usr/bin/env python3
"""
Analyze infer runs and print general statistics.

Outputs aggregated statistics across all specified runs:
- Judgement counts and success rates
- Unparseable responses (llm_score IS NULL)
- Parser issues breakdown
- Time statistics (wall-clock and LLM latency)
- Cost and token usage
- Score distribution
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

from sqlalchemy import func, case
from sqlalchemy.orm import Session

from llm_ensemble.libs.runtime.env import load_runtime_config
from llm_ensemble.libs.db.base import get_engine
from llm_ensemble.libs.db.session import session_context
from llm_ensemble.infer.adapters.driven.io.db.orms import (
    InferRunORM,
    InferRunConfigORM,
    InferRunOutputORM,
    LLMJudgementORM,
    LLMScoreORM,
    ModelConfigORM,
)

# ============================================================================
# CONFIGURATION - Edit these values
# ============================================================================

INCLUDE_RUNS: List[str] = [
    "GPT5-2-100-sample-smoke-test-6",
    "openai-gpt-5-1-100-sample-smoke-test",
    "Claude-Opus-100-sample-smoke-test",
    "ensemble-1-google-gemma-3n-e4b-it",
    "ensemble-1-google-gemma-3-4b-it",
    "ensemble-2-qwen3-8b-",
    "ensemble-2-ministral-3b-2515",
    "ensemble-3-phi-4-multimodal-instruct",
    "ensemble-4-meta-llama-3.2-3b-instruct-start",
    "ensemble-5-ui-tars-1.5-7b-start",
    "ensemble-6-cohere-command-r7b-12-2024-start",
    "reference-ensemble-gpt-5-1-all-samples-start",
    "ensemble-7-qwen3-8b-start",
    "noise-1-ensemble-1-google-gemma-3-4b-it-start",
    "noise-2-ensemble-2-ministral-3b-2512-start",
    "noise-3-ensemble-3-phi-4-multimodal-instruct-start",
    "noise-3-ensemble-3-phi-4-multimodal-instruct-resume-1",
    "noise-4-ensemble-4-meta-llama-3.2-3b-instruct-start",
    "noise-5-ensemble-5-ui-tars-1.5-7b-start",
    "noise-reference-ensemble-gpt-5-1-all-samples-start",
    "+2-noise-reference-ensemble-gpt-5-1-all-samples-start",
    "+2-noise-1-google-gemma-3-4b-it-start",
    "+2-noise-2-ministral-3b-2512",
    "+2-noise-3-phi-4-multimodal-instruct-start",
    "+2-noise-4-meta-llama-3.2-3b-instruct-start",
    "+2-noise-5-ui-tars-1.5-7b-start",
    "+2-noise-3-phi-4-multimodal-instruct-attempt-2",
]

# Set to True to include all runs (ignores INCLUDE_RUNS)
INCLUDE_ALL_RUNS: bool = False

# ============================================================================


@dataclass
class RunStats:
    """Statistics for a collection of infer runs."""

    # Run metadata
    run_count: int = 0
    runs_found: List[str] = None
    runs_not_found: List[str] = None

    # Judgement counts
    total_judgements: int = 0
    unparseable_count: int = 0  # llm_score_id IS NULL
    with_parser_issues: int = 0  # parser_issue_code IS NOT NULL

    # Time
    total_inference_time_ms: float = 0.0  # sum of latency_ms
    avg_latency_ms: float = 0.0

    # Cost
    total_cost_usd: Optional[float] = None

    # Tokens
    total_prompt_tokens: Optional[int] = None
    total_completion_tokens: Optional[int] = None

    # Retries
    total_retries: int = 0
    judgements_with_retries: int = 0

    # Score distribution
    score_distribution: Dict[str, int] = None

    # Parser issues breakdown
    parser_issues: Dict[str, int] = None

    # Models used
    models_used: Dict[str, int] = None

    # Time breakdown by run (run_name -> hours)
    time_by_run: Dict[str, float] = None


def get_basic_stats(session: Session, run_names: List[str], include_all: bool) -> tuple:
    """Get basic judgement statistics."""
    query = (
        session.query(
            func.count(LLMJudgementORM.id).label("total"),
            func.count(case((LLMJudgementORM.llm_score_id.is_(None), 1))).label(
                "unparseable"
            ),
            func.count(case((LLMJudgementORM.parser_issue_code.isnot(None), 1))).label(
                "with_issues"
            ),
            func.sum(LLMJudgementORM.latency_ms).label("total_latency"),
            func.avg(LLMJudgementORM.latency_ms).label("avg_latency"),
            func.sum(
                func.coalesce(
                    LLMJudgementORM.actual_cost_usd, LLMJudgementORM.cost_estimate_usd
                )
            ).label("total_cost"),
            func.sum(LLMJudgementORM.prompt_tokens).label("prompt_tokens"),
            func.sum(LLMJudgementORM.completion_tokens).label("completion_tokens"),
            func.sum(LLMJudgementORM.retries).label("total_retries"),
            func.count(case((LLMJudgementORM.retries > 0, 1))).label("with_retries"),
        )
        .join(
            InferRunOutputORM,
            LLMJudgementORM.infer_run_output_id == InferRunOutputORM.id,
        )
        .join(InferRunORM, InferRunORM.infer_run_output_id == InferRunOutputORM.id)
        .filter(InferRunOutputORM.finished == True)
    )

    if not include_all and run_names:
        query = query.filter(InferRunORM.run_name.in_(run_names))

    return query.one()


def get_found_runs(
    session: Session, run_names: List[str], include_all: bool
) -> List[str]:
    """Get list of run names that were found and finished."""
    query = (
        session.query(InferRunORM.run_name)
        .join(
            InferRunOutputORM, InferRunORM.infer_run_output_id == InferRunOutputORM.id
        )
        .filter(InferRunOutputORM.finished == True)
    )

    if not include_all and run_names:
        query = query.filter(InferRunORM.run_name.in_(run_names))

    return [r[0] for r in query.all()]


def get_score_distribution(
    session: Session, run_names: List[str], include_all: bool
) -> Dict[str, int]:
    """Get distribution of relevance scores."""
    query = (
        session.query(
            LLMScoreORM.label,
            func.count(LLMJudgementORM.id).label("count"),
        )
        .join(LLMJudgementORM, LLMJudgementORM.llm_score_id == LLMScoreORM.id)
        .join(
            InferRunOutputORM,
            LLMJudgementORM.infer_run_output_id == InferRunOutputORM.id,
        )
        .join(InferRunORM, InferRunORM.infer_run_output_id == InferRunOutputORM.id)
        .filter(InferRunOutputORM.finished == True)
    )

    if not include_all and run_names:
        query = query.filter(InferRunORM.run_name.in_(run_names))

    query = query.group_by(LLMScoreORM.label)

    return {
        str(row.label.value) if row.label else "NULL": row.count for row in query.all()
    }


def get_parser_issues(
    session: Session, run_names: List[str], include_all: bool
) -> Dict[str, int]:
    """Get breakdown of parser issues by code."""
    query = (
        session.query(
            LLMJudgementORM.parser_issue_code,
            func.count(LLMJudgementORM.id).label("count"),
        )
        .join(
            InferRunOutputORM,
            LLMJudgementORM.infer_run_output_id == InferRunOutputORM.id,
        )
        .join(InferRunORM, InferRunORM.infer_run_output_id == InferRunOutputORM.id)
        .filter(InferRunOutputORM.finished == True)
        .filter(LLMJudgementORM.parser_issue_code.isnot(None))
    )

    if not include_all and run_names:
        query = query.filter(InferRunORM.run_name.in_(run_names))

    query = query.group_by(LLMJudgementORM.parser_issue_code)

    return {row.parser_issue_code: row.count for row in query.all()}


def get_models_used(
    session: Session, run_names: List[str], include_all: bool
) -> Dict[str, int]:
    """Get count of judgements by model."""
    query = (
        session.query(
            ModelConfigORM.model_id,
            func.count(LLMJudgementORM.id).label("count"),
        )
        .join(
            InferRunOutputORM,
            LLMJudgementORM.infer_run_output_id == InferRunOutputORM.id,
        )
        .join(InferRunORM, InferRunORM.infer_run_output_id == InferRunOutputORM.id)
        .join(
            InferRunConfigORM, InferRunORM.infer_run_config_id == InferRunConfigORM.id
        )
        .join(ModelConfigORM, InferRunConfigORM.model_config_id == ModelConfigORM.id)
        .filter(InferRunOutputORM.finished == True)
    )

    if not include_all and run_names:
        query = query.filter(InferRunORM.run_name.in_(run_names))

    query = query.group_by(ModelConfigORM.model_id)

    return {row.model_id: row.count for row in query.all()}


def collect_stats(
    session: Session, run_names: List[str], include_all: bool
) -> RunStats:
    """Collect all statistics."""
    stats = RunStats()
    stats.runs_found = []
    stats.runs_not_found = []
    stats.score_distribution = {}
    stats.parser_issues = {}
    stats.models_used = {}

    # Find which runs exist
    found = set(get_found_runs(session, run_names, include_all))
    stats.runs_found = sorted(found)
    if not include_all:
        stats.runs_not_found = sorted(set(run_names) - found)

    if not found:
        return stats

    stats.run_count = len(found)

    # Basic stats
    basic = get_basic_stats(session, run_names, include_all)
    stats.total_judgements = basic.total or 0
    stats.unparseable_count = basic.unparseable or 0
    stats.with_parser_issues = basic.with_issues or 0
    stats.total_inference_time_ms = (
        float(basic.total_latency) if basic.total_latency else 0.0
    )
    stats.avg_latency_ms = float(basic.avg_latency) if basic.avg_latency else 0.0
    stats.total_cost_usd = float(basic.total_cost) if basic.total_cost else None
    stats.total_prompt_tokens = (
        int(basic.prompt_tokens) if basic.prompt_tokens else None
    )
    stats.total_completion_tokens = (
        int(basic.completion_tokens) if basic.completion_tokens else None
    )
    stats.total_retries = int(basic.total_retries) if basic.total_retries else 0
    stats.judgements_with_retries = basic.with_retries or 0

    # Distributions
    stats.score_distribution = get_score_distribution(session, run_names, include_all)
    stats.parser_issues = get_parser_issues(session, run_names, include_all)
    stats.models_used = get_models_used(session, run_names, include_all)

    return stats


def format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60
    return f"{hours:.2f}h"


def format_duration_ms(ms: float) -> str:
    """Format milliseconds as human-readable duration."""
    return format_duration(ms / 1000)


def pct(part: int, total: int) -> str:
    """Format percentage."""
    if total == 0:
        return "0.00%"
    return f"{100 * part / total:.2f}%"


def print_statistics(stats: RunStats) -> None:
    """Print formatted statistics."""
    print("=" * 70)
    print("INFER RUN ANALYSIS")
    print("=" * 70)

    # Runs found
    print(f"\nRuns analyzed: {stats.run_count}")
    if stats.runs_not_found:
        print(f"Runs not found: {len(stats.runs_not_found)}")
        for r in stats.runs_not_found[:5]:
            print(f"  - {r}")
        if len(stats.runs_not_found) > 5:
            print(f"  ... and {len(stats.runs_not_found) - 5} more")

    if stats.total_judgements == 0:
        print("\nNo judgements found.")
        return

    # Judgement counts
    print("\n" + "-" * 70)
    print("JUDGEMENTS")
    print("-" * 70)
    print(f"Total judgements:      {stats.total_judgements:,}")
    print(
        f"Unparseable (no score): {stats.unparseable_count:,} ({pct(stats.unparseable_count, stats.total_judgements)})"
    )
    print(
        f"With parser issues:    {stats.with_parser_issues:,} ({pct(stats.with_parser_issues, stats.total_judgements)})"
    )

    # Time
    print("\n" + "-" * 70)
    print("TIME")
    print("-" * 70)
    print(
        f"Total inference time:          {format_duration_ms(stats.total_inference_time_ms)}"
    )
    print(f"Average latency per judgement: {stats.avg_latency_ms:.0f}ms")

    # Cost
    print("\n" + "-" * 70)
    print("COST & TOKENS")
    print("-" * 70)
    if stats.total_cost_usd is not None:
        print(f"Total cost:            ${stats.total_cost_usd:.4f}")
        if stats.total_judgements > 0:
            print(
                f"Average cost/judgement: ${stats.total_cost_usd / stats.total_judgements:.6f}"
            )
    else:
        print("Total cost:            N/A")

    if stats.total_prompt_tokens is not None:
        print(f"Total prompt tokens:   {stats.total_prompt_tokens:,}")
    if stats.total_completion_tokens is not None:
        print(f"Total completion tokens: {stats.total_completion_tokens:,}")
    if stats.total_prompt_tokens and stats.total_completion_tokens:
        total_tokens = stats.total_prompt_tokens + stats.total_completion_tokens
        print(f"Total tokens:          {total_tokens:,}")

    # Retries
    print("\n" + "-" * 70)
    print("RETRIES")
    print("-" * 70)
    print(f"Total retries:         {stats.total_retries:,}")
    print(
        f"Judgements with retries: {stats.judgements_with_retries:,} ({pct(stats.judgements_with_retries, stats.total_judgements)})"
    )

    # Score distribution
    print("\n" + "-" * 70)
    print("SCORE DISTRIBUTION")
    print("-" * 70)
    total_scored = sum(stats.score_distribution.values())
    for label, count in sorted(stats.score_distribution.items()):
        print(f"  {label:20s}: {count:,} ({pct(count, total_scored)})")

    # Parser issues
    if stats.parser_issues:
        print("\n" + "-" * 70)
        print("PARSER ISSUES BREAKDOWN")
        print("-" * 70)
        for code, count in sorted(stats.parser_issues.items(), key=lambda x: -x[1]):
            print(f"  {code:30s}: {count:,}")

    # Models used
    print("\n" + "-" * 70)
    print("MODELS USED")
    print("-" * 70)
    for model, count in sorted(stats.models_used.items(), key=lambda x: -x[1]):
        print(f"  {model:50s}: {count:,} judgements")

    print("\n" + "=" * 70)


def main():
    """Main execution."""
    load_runtime_config()
    engine = get_engine()

    print("Querying database...")

    with session_context(engine) as session:
        stats = collect_stats(session, INCLUDE_RUNS, INCLUDE_ALL_RUNS)

    print_statistics(stats)


if __name__ == "__main__":
    main()
