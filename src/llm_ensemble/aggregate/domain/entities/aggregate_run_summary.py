"""AggregateRunSummary schema - aggregate statistics computed after run completes.

Contains aggregate metrics and statistics computed AFTER the aggregation run finishes.
"""

from __future__ import annotations
from typing import Optional
from pydantic import Field

from llm_ensemble.libs.runtime.run_summary import RunSummary
from llm_ensemble.aggregate.domain.entities.write_summary import WriteSummary


class AggregateRunSummary(RunSummary):
    """Summary of aggregate run with timing and statistics.

    Extends the base RunSummary with aggregation-specific aggregate statistics:
    - Input/output counts (judgements read, unique pairs, aggregated votes)
    - Aggregation statistics (ties, no valid votes)
    - Write summary (persistence statistics from writer port)
    - Warnings summary (counts by warning type)

    This is separate from AggregateRunInfo which contains immutable configuration.
    Written to summary.json for provenance tracking.
    """

    # Core statistics
    input_judgement_count: int = Field(
        ...,
        ge=0,
        description="Total number of input LLMJudgement records read"
    )

    unique_pair_count: int = Field(
        ...,
        ge=0,
        description="Number of unique (query_id, docid) pairs aggregated"
    )

    output_aggregated_count: int = Field(
        ...,
        ge=0,
        description="Number of AggregatedJudgement records written"
    )

    # Aggregation-specific metrics
    tie_count: int = Field(
        default=0,
        ge=0,
        description="Number of times a tie occurred in majority voting"
    )

    no_valid_votes_count: int = Field(
        default=0,
        ge=0,
        description="Number of pairs with no valid votes (all models failed to parse)"
    )

    # Optional warnings summary
    issues_summary: Optional[dict[str, int]] = Field(
        default=None,
        description="Aggregate count of warnings by type (e.g., {'tie': 5, 'no_votes': 2})"
    )

    # Write summary
    write_summary: WriteSummary = Field(
        ...,
        description="Summary of persistence operations from writer port"
    )
