"""Base RunInfo schema - immutable runtime context known before run starts.

This contains all the metadata that is known BEFORE the run begins and remains
immutable throughout execution. This enables attaching run context to individual
domain objects (like LLMJudgement) as soon as they are created, without waiting
for aggregate metrics to be computed at the end.
"""

from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field


class RunInfo(BaseModel):
    """Base runtime context for all CLI runs.

    This captures immutable metadata that is known before the run starts:
    - Run identification (run_id, run_type, cli_name)
    - User-provided context (notes)
    - Git metadata for reproducibility (sha, branch, clean status)

    This class is separate from RunSummary which contains aggregate metrics
    computed AFTER the run completes (timing, counts, statistics).

    CLI-specific run info should extend this class to add configuration metadata.

    Fields in this class are populated by the orchestrator before the domain
    service starts processing, allowing domain objects to embed full provenance
    immediately without waiting for run completion.
    """

    run_id: str = Field(
        ...,
        description="Unique identifier for this run (timestamp-based, e.g., '20250115_143022_gpt-oss-20b')"
    )

    run_type: Literal["official", "test"] = Field(
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

    # Pydantic-specific pattern to make this class immutable
    class Config:
        """Pydantic config."""
        frozen = True  # Make immutable to emphasize this is runtime context
