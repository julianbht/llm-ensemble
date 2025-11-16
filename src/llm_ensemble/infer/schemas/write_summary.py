"""Write summary schema for tracking judgement write operations.

Pure data structure tracking judgement write statistics.
Used as metadata in run summaries for reproducibility and debugging.

For per-write feedback, use WriteResult instead.
"""

from __future__ import annotations
from pydantic import BaseModel, Field


class WriteSummary(BaseModel):
    """Pure data summary of judgement write operations.

    Tracks aggregate write statistics across streaming operations.
    Used as metadata in run summaries - NOT for logging (adapters log directly).

    For per-write feedback with item IDs, use WriteResult instead.
    """

    judgements_written: int = Field(
        default=0,
        ge=0,
        description="Total number of judgements written to disk"
    )
