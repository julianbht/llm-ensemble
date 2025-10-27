"""Base Manifest schema - shared across all CLIs for run metadata."""

from __future__ import annotations
from datetime import datetime
from typing import Literal, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


class Manifest(BaseModel):
    """Base manifest for all CLI runs.

    This captures auto-generated runtime metadata that is common across all CLIs.
    CLI-specific manifests should extend this class to add execution parameters.

    Fields in this class are automatically populated by the orchestrator/run manager.
    """

    run_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this run (auto-generated)"
    )

    run_type: Literal["official", "test"] = Field(
        default="test",
        description="Run type: 'official' for reproducible/git-tracked runs, 'test' for experiments"
    )

    cli_name: str = Field(
        ...,
        description="Name of the CLI that generated this run (e.g., 'ingest', 'infer', 'aggregate', 'evaluate')"
    )

    start_time: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when the run started (auto-captured)"
    )

    end_time: Optional[datetime] = Field(
        default=None,
        description="Timestamp when the run completed (auto-captured at end)"
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

    class Config:
        """Pydantic config."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
