"""Base RunInfo schema - immutable runtime context known before run starts.

This contains all the metadata that is known BEFORE the run begins and remains
immutable throughout execution. This enables attaching run context to individual
domain objects (like LLMJudgement) as soon as they are created, without waiting
for aggregate metrics to be computed at the end.
"""

from __future__ import annotations
from pathlib import Path
from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field

from llm_ensemble.libs.runtime.path_manager import PathManager

class RunType(str, Enum):
    """Enumeration of run types for reproducibility tracking."""
    
    OFFICIAL = "official"
    """Official run: git-tracked, reproducible, for research results."""
    
    TEST = "test"
    """Test run: experimental, may have uncommitted changes."""


class RunInfo(BaseModel):
    """Base runtime context for all CLI runs.

    This captures immutable metadata that is known before the run starts:
    - Run identification (run_name, run_type, cli_name)
    - User-provided context (notes)
    - Git metadata for reproducibility (sha, branch, clean status)

    This class is separate from RunSummary which contains aggregate metrics
    computed AFTER the run completes (timing, counts, statistics).

    CLI-specific run info should extend this class to add configuration metadata.

    Fields in this class are populated by the orchestrator before the domain
    service starts processing, allowing domain objects to embed full provenance
    immediately without waiting for run completion.
    """

    run_name: str = Field(
        ...,
        description="Unique identifier for this run (timestamp-based, e.g., '20250115_143022_gpt-oss-20b')"
    )

    run_type: RunType = Field(
        default="test",
        description="Run type: 'official' for reproducible/git-tracked runs, 'test' for experiments"
    )

    cli_name: str = Field(
        ...,
        description="Name of the CLI that generated this run (e.g., 'ingest', 'infer', 'aggregate', 'evaluate')"
    )

    notes: Optional[str] = Field(
        default=None,
        description="Optional user-provided notes about this run (e.g., experiment purpose)"
    )

    # Git metadata for reproducibility
    git_sha: str = Field(
        ...,
        description="Git commit SHA at time of run (auto-captured)"
    )

    git_clean: bool = Field(
        ...,
        description="Whether git working tree was clean (no uncommitted changes) at time of run"
    )

    git_branch: str = Field(
        ...,
        description="Git branch name at time of run (auto-captured)"
    )

    model_config = ConfigDict(frozen=True)

    @property
    def run_dir(self) -> Path:
        """Derive run directory from run context.

        Computed on-demand from run_name, cli_name, and run_type.
        Single source of truth via PathManager.

        Returns:
            Path to run directory (e.g., artifacts/runs/{cli_name}/{test|official}/{run_name})

        Example:
            >>> run_info = IngestRunInfo(run_name="20250128_120000_test", cli_name="ingest", ...)
            >>> run_info.run_dir
            PosixPath('artifacts/runs/ingest/test/20250128_120000_test')
        """
        return PathManager.get_run_dir(
            cli_name=self.cli_name,
            run_name=self.run_name,
            official=(self.run_type == RunType.OFFICIAL)
        )
