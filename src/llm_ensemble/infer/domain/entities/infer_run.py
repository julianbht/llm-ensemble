"""InferRun - complete record of an inference execution.

Connects configuration (intent) to output (result) with execution metadata.
This is the aggregate root that represents a complete inference run.
"""

from __future__ import annotations
from typing import Optional
from datetime import datetime
from pydantic import Field

from llm_ensemble.libs.runtime.run_info import RunInfo
from llm_ensemble.infer.domain.entities.infer_run_config import InferRunConfig
from llm_ensemble.infer.domain.entities.infer_run_output import InferRunOutput


class InferRun(RunInfo):
    """Complete record of an inference execution.

    Extends RunInfo with inference-specific execution details.

    Aggregate root that connects:
    - What was intended (InferRunConfig) - always present
    - What was produced (InferRunOutput) - set when run completes

    Inherits from RunInfo:
    - id, run_name, run_type
    - notes
    - git_info (git SHA, branch, clean status)

    Lifecycle:
    - Created at start with config, output is None
    - Output added when run completes
    - 1:1 relationship between InferRun and InferRunOutput (same ID)
    """

    cli_name: str = Field(
        default="infer",
        description="Name of the CLI that generated this run (always 'infer')"
    )

    # What was intended (configuration) - always present
    infer_run_config: InferRunConfig = Field(
        ...,
        description="Configuration used for this run"
    )

    # What was produced (output) - None until run completes
    infer_run_output: Optional[InferRunOutput] = Field(
        default=None,
        description="Output produced by this run (None until run completes, then 1:1)"
    )

    # Timing
    start_time: datetime = Field(
        ...,
        description="When the run started"
    )

    end_time: Optional[datetime] = Field(
        default=None,
        description="When the run completed (None until run completes)"
    )
