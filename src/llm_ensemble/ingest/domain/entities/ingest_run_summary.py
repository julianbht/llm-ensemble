"""IngestRunSummary schema - extends base RunSummary with ingest-specific metrics.

This contains ingestion-specific aggregate statistics computed after the run
completes, separate from IngestRunInfo which contains immutable configuration
known before the run starts.
"""

from __future__ import annotations
from pydantic import Field

from llm_ensemble.libs.runtime.run_summary import RunSummary
from llm_ensemble.ingest.domain.entities.write_summary import WriteSummary


class IngestRunSummary(RunSummary):
    """Summary for ingest CLI runs - aggregate metrics computed post-run.

    Extends the base RunSummary with ingestion-specific aggregate statistics:
    - Sample counts (number of samples produced)
    - Write summary (details of what was written to storage)

    This is separate from IngestRunInfo which contains immutable configuration.
    The IngestRunInfo is persisted separately (ingest_run_info.json) to avoid
    duplication. This summary contains only runtime metrics for quick inspection.
    """

    # Aggregate statistics (computed at end of run)
    sample_count: int = Field(
        ...,
        description="Number of judging samples produced"
    )

    write_summary: WriteSummary = Field(
        ...,
        description="Summary of write operations (created vs skipped entities)"
    )
