"""IngestRunInfo schema - extends base RunInfo with ingest-specific configuration.

This contains all ingestion-specific configuration that is known before the run
starts and remains immutable throughout execution. By bundling this with the
base RunInfo, each JudgingSample can carry complete provenance metadata without
waiting for the run to complete.
"""

from __future__ import annotations
from typing import Optional
from pydantic import Field

from llm_ensemble.libs.runtime.run_info import RunInfo
from llm_ensemble.ingest.schemas.ingest_io_config import IngestIOConfig


class IngestRunInfo(RunInfo):
    """Runtime context for ingest CLI runs.

    Extends the base RunInfo with ingest-specific configuration metadata:
    - Which I/O config was used
    - Full configuration object for reproducibility
    - Input parameters (directory path, limit)

    All fields in this class are immutable and known before processing begins,
    allowing JudgingSample objects to embed complete provenance as soon as they
    are created, without waiting for aggregate statistics.

    This is separate from IngestRunSummary which contains post-run metrics like
    sample counts and timing statistics.
    """

    # Override cli_name from base RunInfo to automatically set it to "ingest"
    cli_name: str = Field(
        default="ingest",
        description="Name of the CLI that generated this run (always 'ingest' for IngestRunInfo)"
    )

    # Configuration name (what user requested)
    io_config_name: str = Field(
        ...,
        description="Name of the I/O config used (e.g., 'llm_judge_ingest')"
    )

    # Full configuration object (for reproducibility)
    io_config: IngestIOConfig = Field(
        ...,
        description="I/O configuration used for this run"
    )

    # Input parameters
    input_path: str = Field(
        ...,
        description="Path to input directory containing raw dataset files"
    )

    limit: Optional[int] = Field(
        default=None,
        description="Maximum number of samples to process (None = no limit)"
    )

    # Pydantic-specific pattern to make this class immutable
    class Config:
        """Pydantic config."""
        frozen = True  # Make immutable to emphasize this is runtime context
