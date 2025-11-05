"""Write summary schema for tracking judgement write operations.

This follows the same pattern as ingest's WriteSummary - writers return immutable summaries
that orchestrators can log, rather than writers handling their own logging.
"""

from __future__ import annotations
from typing import Iterator, Dict, Any
from pydantic import BaseModel, Field

from llm_ensemble.libs.logging.log_events import InferWriteEvent


class WriteSummary(BaseModel):
    """Summary of judgement write operations.

    Returned by JudgementWriter implementations to provide transparency
    into what was written during streaming inference.

    This follows the architectural pattern where adapters return summaries
    instead of handling their own logging, maintaining separation of concerns.
    """

    judgements_written: int = Field(default=0, ge=0, description="Number of judgements written")

    def get_log_entries(self) -> Iterator[Dict[str, Any]]:
        """Yield structured log entries for write operations.

        Encapsulates the logging structure within WriteSummary itself,
        so orchestrators don't need to know about internal fields.

        Yields:
            Dict with 'event' key and write statistics

        Example:
            >>> for entry in write_summary.get_log_entries():
            ...     logger.info(**entry)
        """
        # Log judgements written
        if self.judgements_written > 0:
            yield {
                "event": InferWriteEvent.WRITE_JUDGEMENTS,
                "count": self.judgements_written,
            }

        # Always log completion
        yield {
            "event": InferWriteEvent.WRITE_COMPLETE,
            "total_written": self.judgements_written,
        }
