"""Write summary schema for tracking write operations.

Mutable builder for tracking what entities were persisted during write operations.
Used as metadata in run summaries for reproducibility and debugging.
"""

from __future__ import annotations
from pydantic import BaseModel, Field, ConfigDict


class WriteSummary(BaseModel):
    """Incremental builder for tracking write operations.

    Mutable object that tracks what entities were created vs. skipped.
    Writers add to it incrementally as each entity type is persisted.
    Used as metadata in run summaries - NOT for logging (adapters log directly).
    """

    model_config = ConfigDict(validate_assignment=True)

    configs_created: int = Field(default=0, ge=0, description="Number of ingest run configs created")
    configs_skipped: int = Field(default=0, ge=0, description="Number of ingest run configs skipped (already existed)")
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

    def add_configs(self, created: int = 0, skipped: int = 0) -> None:
        """Increment config counts."""
        self.configs_created += created
        self.configs_skipped += skipped

    def add_datasets(self, created: int = 0, skipped: int = 0) -> None:
        """Increment dataset counts."""
        self.datasets_created += created
        self.datasets_skipped += skipped

    def add_runs(self, created: int = 0, skipped: int = 0) -> None:
        """Increment run counts."""
        self.runs_created += created
        self.runs_skipped += skipped

    def add_queries(self, created: int = 0, skipped: int = 0) -> None:
        """Increment query counts."""
        self.queries_created += created
        self.queries_skipped += skipped

    def add_documents(self, created: int = 0, skipped: int = 0) -> None:
        """Increment document counts."""
        self.documents_created += created
        self.documents_skipped += skipped

    def add_samples(self, created: int = 0, skipped: int = 0) -> None:
        """Increment sample counts."""
        self.samples_created += created
        self.samples_skipped += skipped

    @property
    def total_created(self) -> int:
        """Total entities created across all types."""
        return (
            self.configs_created
            + self.datasets_created
            + self.runs_created
            + self.queries_created
            + self.documents_created
            + self.samples_created
        )

    @property
    def total_skipped(self) -> int:
        """Total entities skipped across all types."""
        return (
            self.configs_skipped
            + self.datasets_skipped
            + self.runs_skipped
            + self.queries_skipped
            + self.documents_skipped
            + self.samples_skipped
        )
