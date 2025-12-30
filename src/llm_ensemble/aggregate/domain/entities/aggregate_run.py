"""AggregateRun - complete record of an aggregation execution.

Connects configuration (input) to aggregated dataset (output) with execution metadata.
This is the aggregate root that represents a complete aggregation run.
"""

from __future__ import annotations
from datetime import datetime
from pydantic import ConfigDict, Field

from llm_ensemble.libs.runtime.run_info import RunInfo
from llm_ensemble.aggregate.domain.entities.aggregate_run_config import AggregateRunConfig
from llm_ensemble.aggregate.domain.entities.aggregated_dataset import AggregatedDataset


class AggregateRun(RunInfo):
    """Complete record of an aggregation execution.

    Extends RunInfo with aggregate-specific execution details.

    Aggregate root that connects:
    - What was intended (AggregateRunConfig)
    - What was produced (AggregatedDataset)
    - When it was executed (timing)

    Inherits from RunInfo:
    - id, run_name, run_type
    - notes
    - git_info (git SHA, branch, clean status)
    """

    cli_name: str = Field(
        default="aggregate",
        description="Name of the CLI that generated this run (always 'aggregate')"
    )

    # What was intended (configuration)
    aggregate_run_config: AggregateRunConfig = Field(
        ...,
        description="Configuration used for this run"
    )

    # What was produced (output)
    aggregated_dataset: AggregatedDataset = Field(
        ...,
        description="Dataset produced by this run"
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

    model_config = ConfigDict(frozen=True)
