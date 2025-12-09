"""IngestRunInfo schema - extends base RunInfo with ingest-specific configuration.

This contains all ingestion-specific configuration that is known before the run
starts and remains immutable throughout execution. By bundling this with the
base RunInfo, each JudgingSample can carry complete provenance metadata without
waiting for the run to complete.
"""

from __future__ import annotations
from typing import Optional
from uuid import UUID, uuid4
from pydantic import ConfigDict, Field

from llm_ensemble.libs.runtime.run_info import RunInfo
from llm_ensemble.libs.schemas import IOConfig


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

    The id field is a random UUID (v4).
    """

    # Random UUID
    id: UUID = Field(
        default_factory=uuid4,
        description="Random UUID identifier"
    )

    # Override cli_name from base RunInfo to automatically set it to "ingest"
    cli_name: str = Field(
        default="ingest",
        description="Name of the CLI that generated this run (always 'ingest' for IngestRunInfo)"
    )

    # Configuration name 
    io_config_name: str = Field(
        ...,
        description="Name of the I/O config used (e.g., 'llm_judge_challenge_json')"
    )

    # Full configuration object (for reproducibility)
    io_config: IOConfig = Field(
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

    model_config = ConfigDict(frozen=True)
    
    @classmethod
    def create(
        cls,
        run_name: str,
        io_config_name: str,
        io_config: IOConfig,
        input_path: str,
        limit: Optional[int] = None,
        **kwargs
    ) -> "IngestRunInfo":
        """Create an IngestRunInfo with random UUID.

        Args:
            run_name: Run identifier (timestamp-based)
            io_config_name: I/O config name
            io_config: Full I/O configuration
            input_path: Input directory path
            limit: Optional sample limit
            **kwargs: Additional fields from base RunInfo (git_sha, etc.)

        Returns:
            IngestRunInfo instance with random UUID
        """
        return cls(
            run_name=run_name,
            io_config_name=io_config_name,
            io_config=io_config,
            input_path=input_path,
            limit=limit,
            **kwargs
        )
