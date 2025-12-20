"""InferRunInfo schema - runtime metadata for infer CLI runs.

Contains only run metadata (git info, timestamps, notes, run_type).
Configuration and execution context are separated into:
- InferRunConfig: Model/adapter/retry configuration
- InferRunContext: CLI args (input_run_name, start_idx, end_idx, io_name)
- InferRunOutput: Judgements and aggregate metrics produced

Separation of concerns:
- InferRunInfo: Run metadata (git SHA, timestamps, run_type, notes)
- InferRunConfig: Configuration used to produce judgements
- InferRunContext: Execution context (input source, sample range)
- InferRunOutput: Actual judgements and metrics produced
"""

from __future__ import annotations
from pydantic import ConfigDict, Field

from llm_ensemble.libs.runtime.run_info import RunInfo


class InferRunInfo(RunInfo):
    """Runtime metadata for infer CLI runs.

    Pure Pydantic model with no methods - just data.
    Contains only run metadata inherited from RunInfo:
    - Run identification (id, run_name, cli_name)
    - Run type (official vs test)
    - User context (notes)
    - Git metadata (commit SHA, branch, clean status)
    - Timestamps (start_time, end_time via RunInfo)

    Configuration and execution context are separated into InferRunConfig and
    InferRunContext respectively. This keeps concerns cleanly separated.
    """

    cli_name: str = Field(
        default="infer",
        description="Name of the CLI that generated this run (always 'infer' for InferRunInfo)"
    )

    model_config = ConfigDict(frozen=True)
