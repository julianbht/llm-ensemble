"""IngestRunInfo schema - runtime metadata for ingest CLI runs.

Contains only run metadata (git info, timestamps, notes, run_type).
Configuration and execution context are separated into IngestRunConfig.

Separation of concerns:
- IngestRunInfo: Run metadata (git SHA, timestamps, run_type, notes)
- IngestRunConfig: Configuration used to process the dataset (I/O config, input path, limit)
"""

from __future__ import annotations
from pydantic import ConfigDict, Field

from llm_ensemble.libs.runtime.run_info import RunInfo


class IngestRunInfo(RunInfo):
    """Runtime metadata for ingest CLI runs.

    Pure Pydantic model with no methods - just data.
    Contains only run metadata inherited from RunInfo:
    - Run identification (id, run_name, cli_name)
    - Run type (official vs test)
    - User context (notes)
    - Git metadata (commit SHA, branch, clean status)
    - Timestamps (start_time, end_time via RunInfo)

    Configuration and execution context are separated into IngestRunConfig.
    This keeps concerns cleanly separated.
    """

    cli_name: str = Field(
        default="ingest",
        description="Name of the CLI that generated this run (always 'ingest' for IngestRunInfo)"
    )

    model_config = ConfigDict(frozen=True)
