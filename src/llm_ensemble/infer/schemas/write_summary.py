"""Write summary schema for tracking judgement write operations.

This follows the same pattern as the ingest WriteSummary - writers return
immutable summaries that orchestrators can log, rather than writers handling
their own logging (separation of concerns).

Unlike the ingest CLI which does batch inserts of multiple entity types,
the infer CLI writes judgements one at a time. This summary simply tracks
that a judgement was successfully written to disk, along with the sample ID
for traceability.
"""

from __future__ import annotations
from typing import Iterator, Dict, Any, Optional
from uuid import UUID
from pydantic import BaseModel, Field

from llm_ensemble.libs.logging.log_events import InferWriteEvent


class WriteSummary(BaseModel):
    """Summary of judgement write operations.

    Returned by JudgementWriter implementations to provide transparency
    into write operations during streaming inference.

    This follows the architectural pattern where adapters return summaries
    instead of handling their own logging, maintaining separation of concerns.

    Unlike the ingest WriteSummary which tracks batch operations across multiple
    entity types (datasets, runs, queries, documents, samples), this tracks
    individual judgement writes with the sample ID for traceability.
    """

    judgements_written: int = Field(
        default=0,
        ge=0,
        description="Number of judgements written to disk"
    )

    sample_id: Optional[UUID] = Field(
        default=None,
        description="ID of the sample that was written (for per-item write tracking)"
    )

    def get_log_entries(self) -> Iterator[Dict[str, Any]]:
        """Yield structured log entries for write operations.

        Encapsulates the logging structure within WriteSummary itself,
        so orchestrators don't need to know about internal fields.

        Yields:
            Dict with 'event' key and judgement-specific counts

        Example:
            >>> for entry in write_summary.get_log_entries():
            ...     logger.info(**entry)
        """
        # Only log if judgements were written
        if self.judgements_written > 0:
            log_entry = {
                "event": InferWriteEvent.WRITE_JUDGEMENT_COMPLETE,
                "judgements_written": self.judgements_written,
            }
            # Add sample_id if present (for per-item writes)
            if self.sample_id is not None:
                log_entry["sample_id"] = str(self.sample_id)
            yield log_entry
