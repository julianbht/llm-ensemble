"""IngestRun - complete record of an ingest execution.

Connects configuration (input) to dataset (output) with execution metadata.
This is the aggregate root that represents a complete ingest run.
"""

from __future__ import annotations
from typing import Optional
from datetime import datetime
from pydantic import Field

from llm_ensemble.libs.runtime.run_info import RunInfo
from llm_ensemble.ingest.domain.entities.ingest_run_config import IngestRunConfig
from llm_ensemble.ingest.domain.entities.normalized_dataset import NormalizedDataset


class IngestRun(RunInfo):
    """Complete record of an ingest execution.

    Extends RunInfo with ingest-specific execution details.

    Aggregate root that connects:
    - What was intended (IngestRunConfig)
    - What was produced (NormalizedDataset)
    - When it was executed (timing)

    Inherits from RunInfo:
    - id, run_name, run_type
    - notes
    - git_info (git SHA, branch, clean status)
    """

    cli_name: str = Field(
        default="ingest",
        description="Name of the CLI that generated this run (always 'ingest')"
    )

    # What was intended (configuration)
    ingest_run_config: IngestRunConfig = Field(
        ...,
        description="Configuration used for this run"
    )

    # What was produced (output)
    normalized_dataset: Optional[NormalizedDataset] = Field(
        ...,
        description="Dataset produced by this run (None when included in summary to avoid duplication)"
    )

    # Timing
    start_time: datetime = Field(
        ...,
        description="When the run started"
    )

    end_time: datetime = Field(
        ...,
        description="When the run completed"
    )
