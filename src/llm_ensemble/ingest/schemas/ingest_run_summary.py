"""IngestRunSummary schema - extends base RunSummary with ingest-specific metrics.

This contains ingestion-specific aggregate statistics computed after the run
completes, separate from IngestRunInfo which contains immutable configuration
known before the run starts.
"""

from __future__ import annotations
from pydantic import Field

from llm_ensemble.libs.runtime.run_summary import RunSummary
from llm_ensemble.ingest.schemas.ingest_run_info import IngestRunInfo
from llm_ensemble.ingest.schemas.write_summary import WriteSummary


class IngestRunSummary(RunSummary):
    """Summary for ingest CLI runs - aggregate metrics computed post-run.

    Extends the base RunSummary with ingestion-specific aggregate statistics:
    - Sample counts (number of samples produced)
    - Write summary (details of what was written to storage)

    This is separate from IngestRunInfo which contains immutable configuration.
    The IngestRunInfo can be embedded in JudgingSample objects immediately, while
    IngestRunSummary is only computed and written at the end of the run.

    The summary includes the full IngestRunInfo, so it contains both the runtime
    context (I/O config, input path, etc.) and the post-run metrics.
    """

    # Override to use IngestRunInfo instead of base RunInfo
    run_info: IngestRunInfo = Field(
        ...,
        description="Immutable ingestion run context (I/O config, input path, etc.)"
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
