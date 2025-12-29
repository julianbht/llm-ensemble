"""IngestRun - complete record of an ingest execution.

Connects configuration (input) to dataset (output) with execution metadata.
This is the aggregate root that represents a complete ingest run.
"""

from __future__ import annotations
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, ConfigDict, Field

from llm_ensemble.libs.runtime.run_info import RunType
from llm_ensemble.ingest.domain.entities.ingest_run_config import IngestRunConfig
from llm_ensemble.ingest.domain.entities.normalized_dataset import NormalizedDataset


class IngestRun(BaseModel):
    """Complete record of an ingest execution.

    Aggregate root that connects:
    - What was intended (IngestRunConfig)
    - What was produced (NormalizedDataset)
    - When and how it was executed (timing, git metadata)

    This matches the IngestRunORM structure and represents the complete
    execution record.
    """

    id: UUID = Field(
        default_factory=uuid4,
        description="Random UUID identifier for this run"
    )

    run_name: str = Field(
        ...,
        description="Unique identifier for this run (timestamp-based)"
    )

    run_type: RunType = Field(
        default=RunType.TEST,
        description="Run type: 'official' for reproducible runs, 'test' for experiments"
    )

    # What was intended (configuration)
    ingest_run_config: IngestRunConfig = Field(
        ...,
        description="Configuration used for this run"
    )

    # What was produced (output)
    normalized_dataset: NormalizedDataset = Field(
        ...,
        description="Dataset produced by this run"
    )

    # Timing
    start_time: datetime = Field(
        ...,
        description="When the run started"
    )

    end_time: datetime = Field(
        ...,
        description="When the run completed"
    )

    # Git metadata for reproducibility
    git_sha: str = Field(
        ...,
        description="Git commit SHA at time of run"
    )

    git_branch: str = Field(
        ...,
        description="Git branch at time of run"
    )

    git_is_dirty: str = Field(
        ...,
        description="Whether git working directory was clean ('true' or 'false')"
    )

    notes: Optional[str] = Field(
        default=None,
        description="Optional user-provided notes about this run"
    )

    model_config = ConfigDict(frozen=True)
