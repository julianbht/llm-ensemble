"""Base RunSummary schema - aggregate metrics computed after run completion.

This contains statistics and metrics that are only known AFTER the run completes.
This is separate from RunInfo which contains immutable context known before the
run starts. By splitting these concerns, domain objects can embed RunInfo for
provenance without waiting for aggregate metrics.
"""

from __future__ import annotations
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

from llm_ensemble.libs.runtime.run_info import RunInfo


class RunSummary(BaseModel):
    """Base summary for all CLI runs - aggregate metrics computed post-run.

    This captures metrics and statistics that are only known after the run
    completes:
    - Timing (start_time, end_time, duration)
    - Aggregate statistics (counts, averages, summaries)

    This is separate from RunInfo (which contains immutable runtime context).
    The RunInfo can be embedded in domain objects immediately, while RunSummary
    is only computed and written at the end of the run.

    CLI-specific summaries should extend this class to add domain-specific metrics.

    The RunSummary includes the full RunInfo for convenience, so the summary
    contains both the runtime context and the post-run metrics in one place.
    """

    # Embed full run info for convenience (summary contains both context + metrics)
    run_info: RunInfo = Field(
        ...,
        description="Immutable runtime context (known before run starts)"
    )

    # Timing (computed during/after run)
    start_time: datetime = Field(
        ...,
        description="Timestamp when the run started (captured at beginning of domain service execution)"
    )

    end_time: datetime = Field(
        ...,
        description="Timestamp when the run completed (captured at end)"
    )

    class Config:
        """Pydantic config."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
