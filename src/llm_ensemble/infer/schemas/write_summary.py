"""Write summary schema for tracking judgement write operations.

This follows the same pattern as the ingest WriteSummary - writers return
immutable summaries that orchestrators can log, rather than writers handling
their own logging (separation of concerns).

Unlike the ingest CLI which does batch inserts of multiple entity types,
the infer CLI writes judgements one at a time. This summary tracks aggregate
statistics across all write operations (returned after streaming completes).

For per-write feedback, use WriteResult instead.
"""

from __future__ import annotations
from typing import Iterator, Dict, Any
from pydantic import BaseModel, Field

from llm_ensemble.libs.logging.log_events import InferWriteEvent


class WriteSummary(BaseModel):
    """Aggregate summary of judgement write operations.

    Returned by JudgementWriter implementations to provide transparency
    into write operations during streaming inference.

    This follows the architectural pattern where adapters return summaries
    instead of handling their own logging, maintaining separation of concerns.

    Unlike the ingest WriteSummary which tracks batch operations across multiple
    entity types (datasets, runs, queries, documents, samples), this tracks
    aggregate judgement writes across all streaming operations.

    For per-write feedback with item IDs, use WriteResult instead.
    """

    judgements_written: int = Field(
        default=0,
        ge=0,
        description="Total number of judgements written to disk"
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
            yield {
                "event": InferWriteEvent.WRITE_COMPLETE,
                "total_judgements": self.judgements_written,
            }
