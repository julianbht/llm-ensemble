"""Write result schema for tracking individual write operations.

This schema represents the result of a single write operation, providing
immediate feedback for logging and traceability during streaming writes.

Distinct from WriteSummary which represents aggregate statistics across
multiple write operations.
"""

from __future__ import annotations
from uuid import UUID
from pydantic import BaseModel, Field


class WriteResult(BaseModel):
    """Result of writing a single item to storage.

    Provides immediate feedback for individual write operations.
    Used for per-item logging and traceability during streaming.

    This is distinct from WriteSummary:
    - WriteResult: Per-operation feedback (e.g., "wrote judgement X")
    - WriteSummary: Aggregate statistics (e.g., "wrote 100 judgements total")
    """

    item_id: UUID = Field(
        description="ID of the item that was written"
    )

    item_type: str = Field(
        description="Type of item written (e.g., 'judgement', 'sample')"
    )
