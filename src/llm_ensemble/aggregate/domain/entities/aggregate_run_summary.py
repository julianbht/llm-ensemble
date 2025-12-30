"""AggregateRunSummary schema - aggregate statistics computed after run completes.

Contains aggregate metrics and statistics computed AFTER the aggregation run finishes.
"""

from __future__ import annotations
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

from llm_ensemble.aggregate.domain.entities.write_summary import WriteSummary


class AggregateRunSummary(BaseModel):
    """Summary of aggregate run with timing and statistics.

    Contains:
    - Timing: start_time, end_time
    - Statistics: counts, aggregation metrics
    - Metadata: warnings summary, tie statistics
    - Write summary: persistence statistics from writer port

    This is the final summary object created after the run completes.
    Written to summary.json for provenance tracking.
    """

    start_time: datetime = Field(
        ...,
        description="When aggregation processing started"
    )

    end_time: datetime = Field(
        ...,
        description="When aggregation processing completed"
    )

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
    warnings_summary: Optional[dict[str, int]] = Field(
        default=None,
        description="Aggregate count of warnings by type (e.g., {'tie': 5, 'no_votes': 2})"
    )

    # Write summary
    write_summary: WriteSummary = Field(
        ...,
        description="Summary of persistence operations from writer port"
    )
