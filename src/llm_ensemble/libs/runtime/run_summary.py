"""Base RunSummary schema - aggregate metrics computed after run completion.

This contains statistics and metrics that are only known AFTER the run completes.
This is separate from RunInfo which contains immutable context known before the
run starts. By splitting these concerns, domain objects can embed RunInfo for
provenance without waiting for aggregate metrics.
"""

from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel, Field


class RunSummary(BaseModel):
    """Base summary for all CLI runs - aggregate metrics computed post-run.

    This captures metrics and statistics that are only known after the run
    completes:
    - Timing (start_time, end_time, duration)
    - Aggregate statistics (counts, averages, summaries)

    This is separate from RunInfo (which contains immutable runtime context).
    The RunInfo is persisted separately (e.g., infer_run_info.json) to avoid
    duplication. The summary contains only runtime metrics for quick inspection.

    CLI-specific summaries should extend this class to add domain-specific metrics.
    """

    # Timing (computed during/after run)
    start_time: datetime = Field(
        ...,
        description="Timestamp when the run started (captured at beginning of domain service execution)"
    )

    end_time: datetime = Field(
        ...,
        description="Timestamp when the run completed (captured at end)"
    )
