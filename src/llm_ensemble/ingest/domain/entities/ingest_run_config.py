"""IngestRunConfig - immutable configuration bundle for an ingest run.

Pure Pydantic model containing all configuration needed to execute ingestion:
- I/O configuration (which dataset adapter to use)
- Input source (where the raw data comes from)
- Processing limits (how many samples to process)

This is a serializable domain entity (frozen Pydantic model).
Separate from IngestRunInfo (git info, timestamps, run metadata).
"""

from __future__ import annotations
from typing import Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, ConfigDict, Field


class IngestRunConfig(BaseModel):
    """Immutable configuration bundle for an ingest run.

    Pure Pydantic model - no business logic, just data.
    Contains configuration decisions made for this specific ingestion run.
    """

    id: UUID = Field(
        default_factory=uuid4,
        description="Random UUID for this config bundle"
    )

    io_config_name: str = Field(
        ...,
        description="Name of the I/O config used (e.g., 'llm_judge_ingest')"
    )

    input_path: str = Field(
        ...,
        description="Path to input directory containing raw dataset files"
    )

    limit: Optional[int] = Field(
        default=None,
        description="Maximum number of samples to process (None = no limit)"
    )

    model_config = ConfigDict(frozen=True)
