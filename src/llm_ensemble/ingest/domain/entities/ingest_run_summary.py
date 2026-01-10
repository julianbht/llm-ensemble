"""IngestRunSummary schema - extends base RunSummary with ingest-specific metrics.

This contains ingestion-specific aggregate statistics computed after the run
completes, separate from IngestRunInfo which contains immutable configuration
known before the run starts.
"""

from __future__ import annotations
from pydantic import Field

from llm_ensemble.libs.runtime.run_summary import RunSummary
from llm_ensemble.ingest.domain.entities.write_summary import WriteSummary
from llm_ensemble.ingest.domain.entities.ingest_run import IngestRun


class IngestRunSummary(RunSummary):
    """Summary for ingest CLI runs - aggregate metrics computed post-run.

    Extends the base RunSummary with ingestion-specific aggregate statistics:
    - Run: complete run entity (includes config, git info, timestamps, run metadata)
    - Sample counts (number of samples produced)
    - Write summary (details of what was written to storage)
    """

    run: IngestRun = Field(
        ...,
        description="Complete ingest run entity (includes config, git info, timestamps, run metadata)"
    )

    # Aggregate statistics (computed at end of run)
    sample_count: int = Field(
        ...,
        description="Number of judging samples produced"
    )

    write_summary: WriteSummary = Field(
        ...,
        description="Summary of write operations (created vs skipped entities)"
    )
