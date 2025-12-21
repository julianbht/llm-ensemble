"""Base RunInfo schema - immutable runtime context known before run starts.

This contains all the metadata that is known BEFORE the run begins and remains
immutable throughout execution. This enables attaching run context to individual
domain objects (like LLMJudgement) as soon as they are created, without waiting
for aggregate metrics to be computed at the end.
"""

from __future__ import annotations
from pathlib import Path
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, ConfigDict, Field

from llm_ensemble.libs.runtime.path_manager import PathManager
from llm_ensemble.libs.runtime.git_utils import GitInfo, get_git_info

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
    - Git metadata for reproducibility (git_info)

    This class is separate from RunSummary which contains aggregate metrics
    computed AFTER the run completes (timing, counts, statistics).

    CLI-specific run info should extend this class to add configuration metadata.

    Fields in this class are populated by the orchestrator before the domain
    service starts processing, allowing domain objects to embed full provenance
    immediately without waiting for run completion.
    """

    id: UUID = Field(
        default_factory=uuid4,
        description="Random UUID identifier for this run"
    )

    run_name: str = Field(
        ...,
        description="Unique identifier for this run (timestamp-based, e.g., '20250115_143022_gpt-oss-20b')"
    )

    run_type: RunType = Field(
        default=RunType.TEST,
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
    git_info: GitInfo = Field(
        default_factory=get_git_info,
        description="Git metadata (commit SHA, branch, clean status) captured at time of run"
    )

    model_config = ConfigDict(frozen=True)

    @property
    def run_dir(self) -> Path:
        """Derive run directory from run context.

        Computed on-demand from run_name, cli_name, and run_type.
        Single source of truth via PathManager.

        Returns:
            Path to run directory (e.g., artifacts/runs/{cli_name}/{test|official}/{run_name})
        """
        return PathManager.get_run_dir(
            cli_name=self.cli_name,
            run_name=self.run_name,
            official=(self.run_type == RunType.OFFICIAL)
        )
