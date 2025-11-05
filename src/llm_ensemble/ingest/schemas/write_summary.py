"""Write summary schema for tracking database write operations.

This follows the same pattern as RunSummary - writers return immutable summaries
that orchestrators can log, rather than writers handling their own logging.
"""

from __future__ import annotations
from typing import Iterator, Dict, Any
from pydantic import BaseModel, Field

from llm_ensemble.ingest.log_events import IngestWriteEvent


class WriteSummary(BaseModel):
    """Summary of database write operations.

    Returned by DatasetWriter implementations to provide transparency
    into what was created vs. skipped during idempotent writes.

    This follows the architectural pattern where adapters return summaries
    instead of handling their own logging, maintaining separation of concerns.
    """

    datasets_created: int = Field(default=0, ge=0, description="Number of datasets created")
    datasets_skipped: int = Field(default=0, ge=0, description="Number of datasets skipped (already existed)")
    runs_created: int = Field(default=0, ge=0, description="Number of ingest runs created")
    runs_skipped: int = Field(default=0, ge=0, description="Number of ingest runs skipped (already existed)")
    queries_created: int = Field(default=0, ge=0, description="Number of queries created")
    queries_skipped: int = Field(default=0, ge=0, description="Number of queries skipped (already existed)")
    documents_created: int = Field(default=0, ge=0, description="Number of documents created")
    documents_skipped: int = Field(default=0, ge=0, description="Number of documents skipped (already existed)")
    samples_created: int = Field(default=0, ge=0, description="Number of samples created")
    samples_skipped: int = Field(default=0, ge=0, description="Number of samples skipped (already existed)")

    @property
    def total_created(self) -> int:
        """Total entities created across all types."""
        return (
            self.datasets_created
            + self.runs_created
            + self.queries_created
            + self.documents_created
            + self.samples_created
        )

    @property
    def total_skipped(self) -> int:
        """Total entities skipped across all types."""
        return (
            self.datasets_skipped
            + self.runs_skipped
            + self.queries_skipped
            + self.documents_skipped
            + self.samples_skipped
        )

    def get_log_entries(self) -> Iterator[Dict[str, Any]]:
        """Yield structured log entries for each entity type with activity.

        Encapsulates the logging structure within WriteSummary itself,
        so orchestrators don't need to know about internal fields.

        Yields:
            Dict with 'event' key and entity-specific created/skipped counts

        Example:
            >>> for entry in write_summary.get_log_entries():
            ...     logger.info(**entry)
        """
        # Only log entity types that had activity
        if self.datasets_created > 0 or self.datasets_skipped > 0:
            yield {
                "event": IngestWriteEvent.WRITE_DATASETS,
                "created": self.datasets_created,
                "skipped": self.datasets_skipped,
            }

        if self.runs_created > 0 or self.runs_skipped > 0:
            yield {
                "event": IngestWriteEvent.WRITE_RUNS,
                "created": self.runs_created,
                "skipped": self.runs_skipped,
            }

        if self.queries_created > 0 or self.queries_skipped > 0:
            yield {
                "event": IngestWriteEvent.WRITE_QUERIES,
                "created": self.queries_created,
                "skipped": self.queries_skipped,
            }

        if self.documents_created > 0 or self.documents_skipped > 0:
            yield {
                "event": IngestWriteEvent.WRITE_DOCUMENTS,
                "created": self.documents_created,
                "skipped": self.documents_skipped,
            }

        if self.samples_created > 0 or self.samples_skipped > 0:
            yield {
                "event": IngestWriteEvent.WRITE_JUDGING_SAMPLES,
                "created": self.samples_created,
                "skipped": self.samples_skipped,
            }

        # Always log totals if there was any activity
        if self.total_created > 0 or self.total_skipped > 0:
            yield {
                "event": IngestWriteEvent.WRITE_COMPLETE,
                "total_created": self.total_created,
                "total_skipped": self.total_skipped,
            }
