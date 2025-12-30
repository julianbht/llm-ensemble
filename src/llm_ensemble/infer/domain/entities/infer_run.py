"""InferRun - complete record of an inference execution.

Connects configuration (intent) to output (result) with execution metadata.
This is the aggregate root that represents a complete inference run.
"""

from __future__ import annotations
from pydantic import ConfigDict, Field

from llm_ensemble.libs.runtime.run_info import RunInfo
from llm_ensemble.infer.domain.entities.infer_run_config import InferRunConfig
from llm_ensemble.infer.domain.entities.infer_run_output import InferRunOutput


class InferRun(RunInfo):
    """Complete record of an inference execution.

    Extends RunInfo with inference-specific execution details.

    Aggregate root that connects:
    - What was intended (InferRunConfig)
    - What was produced (InferRunOutput)

    Inherits from RunInfo:
    - id, run_name, run_type
    - notes
    - git_info (git SHA, branch, clean status)

    1:1 relationship between InferRun and InferRunOutput (same ID).
    """

    cli_name: str = Field(
        default="infer",
        description="Name of the CLI that generated this run (always 'infer')"
    )

    # What was intended (configuration)
    infer_run_config: InferRunConfig = Field(
        ...,
        description="Configuration used for this run"
    )

    # What was produced (output) - 1:1 relationship
    infer_run_output: InferRunOutput = Field(
        ...,
        description="Output produced by this run (1:1, always present)"
    )

    model_config = ConfigDict(frozen=True)
