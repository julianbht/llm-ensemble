"""EvaluateRun - complete record of an evaluation execution.

Connects configuration (input) to metric results (output) with execution metadata.
This is the aggregate root that represents a complete evaluation run.
"""

from __future__ import annotations
from datetime import datetime
from pydantic import Field

from llm_ensemble.libs.runtime.run_info import RunInfo
from llm_ensemble.evaluate.domain.entities.evaluate_run_config import EvaluateRunConfig
from llm_ensemble.evaluate.domain.entities.metric_result import MetricResult


class EvaluateRun(RunInfo):
    """Complete record of an evaluation execution.

    Extends RunInfo with evaluate-specific execution details.

    Aggregate root that connects:
    - What was intended (EvaluateRunConfig)
    - What was produced (metric_results)
    - When it was executed (timing)

    Inherits from RunInfo:
    - id, run_name, run_type
    - notes
    - git_info (git SHA, branch, clean status)
    """

    cli_name: str = Field(
        default="evaluate",
        description="Name of the CLI that generated this run (always 'evaluate')"
    )

    # What was intended (configuration)
    evaluate_run_config: EvaluateRunConfig = Field(
        ...,
        description="Configuration used for this run"
    )

    # What was produced (output)
    metric_results: list[MetricResult] = Field(
        ...,
        description="Metrics computed by this run"
    )

    # Metadata about the evaluated run
    evaluated_run_type: str = Field(
        ...,
        description="Type of run that was evaluated ('infer' or 'aggregate')"
    )

    evaluated_sample_count: int = Field(
        ...,
        ge=0,
        description="Number of samples in the evaluated dataset"
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
