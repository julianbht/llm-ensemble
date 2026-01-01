"""Base RunSummary schema - aggregate metrics computed after run completion.

This contains statistics and metrics that are only known AFTER the run completes.
This is separate from RunInfo which contains immutable context known before the
run starts. By splitting these concerns, domain objects can embed RunInfo for
provenance without waiting for aggregate metrics.
"""

from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel, Field, computed_field


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

    start_time: datetime = Field(
        ...,
        description="Timestamp when the run started (captured at beginning of domain service execution)"
    )

    end_time: datetime = Field(
        ...,
        description="Timestamp when the run completed (captured at end)"
    )

    @computed_field
    @property
    def duration_seconds(self) -> float:
        """Duration of the run in seconds (precise measurement)."""
        return (self.end_time - self.start_time).total_seconds()

    @computed_field
    @property
    def duration_minutes(self) -> float:
        """Duration of the run in minutes (for medium-length runs)."""
        return self.duration_seconds / 60.0

    @computed_field
    @property
    def duration_hours(self) -> float:
        """Duration of the run in hours (for long-running jobs)."""
        return self.duration_seconds / 3600.0
