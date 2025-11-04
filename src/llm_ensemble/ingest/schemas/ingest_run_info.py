"""IngestRunInfo schema - extends base RunInfo with ingest-specific configuration.

This contains all ingestion-specific configuration that is known before the run
starts and remains immutable throughout execution. By bundling this with the
base RunInfo, each JudgingSample can carry complete provenance metadata without
waiting for the run to complete.
"""

from __future__ import annotations
from typing import Optional
from uuid import UUID
from pydantic import Field

from llm_ensemble.libs.runtime.run_info import RunInfo
from llm_ensemble.ingest.schemas.ingest_io_config import IngestIOConfig
from llm_ensemble.libs.db import compute_ingest_run_uuid


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
    
    The id field is a mandatory deterministic UUID computed from run_name.
    """

    # Deterministic UUID
    id: UUID = Field(
        ...,
        description="Deterministic UUID computed from run_name"
    )

    # Override cli_name from base RunInfo to automatically set it to "ingest"
    cli_name: str = Field(
        default="ingest",
        description="Name of the CLI that generated this run (always 'ingest' for IngestRunInfo)"
    )

    # Configuration name (what user requested)
    io_config_name: str = Field(
        ...,
        description="Name of the I/O config used (e.g., 'llm_judge_challenge_ndjson')"
    )

    # Full configuration object (for reproducibility) - now uses IngestIOConfig
    io_config: IngestIOConfig = Field(
        ...,
        description="Ingest-specific I/O configuration used for this run"
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
    
    @classmethod
    def create(
        cls,
        run_name: str,
        io_config_name: str,
        io_config: IngestIOConfig,
        input_path: str,
        limit: Optional[int] = None,
        **kwargs
    ) -> "IngestRunInfo":
        """Create an IngestRunInfo with computed deterministic UUID.
        
        Args:
            run_name: Run identifier (timestamp-based)
            io_config_name: I/O config name
            io_config: Full I/O configuration
            input_path: Input directory path
            limit: Optional sample limit
            **kwargs: Additional fields from base RunInfo (git_sha, etc.)
        
        Returns:
            IngestRunInfo instance with computed id
        
        Example:
            >>> run_info = IngestRunInfo.create(
            ...     run_name="20250128_120000_abc123",
            ...     io_config_name="llm_judge_challenge_ndjson",
            ...     io_config=config,
            ...     input_path="/data/llmjudge"
            ... )
        """
        run_info_id = compute_ingest_run_uuid(run_name)
        return cls(
            id=run_info_id,
            run_name=run_name,
            io_config_name=io_config_name,
            io_config=io_config,
            input_path=input_path,
            limit=limit,
            **kwargs
        )
