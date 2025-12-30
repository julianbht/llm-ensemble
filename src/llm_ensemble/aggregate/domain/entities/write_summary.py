"""Write summary schema for tracking aggregate write operations.

Mutable builder for tracking what entities were persisted during aggregation writes.
Used as metadata in run summaries for reproducibility and debugging.
"""

from __future__ import annotations
from pydantic import BaseModel, Field, ConfigDict


class WriteSummary(BaseModel):
    """Incremental builder for tracking aggregate write operations.

    Mutable object that tracks what entities were created vs. skipped.
    Writers add to it incrementally as each entity type is persisted.
    Used as metadata in run summaries.
    """

    model_config = ConfigDict(validate_assignment=True)

    # Run metadata (created once during open)
    aggregation_specs_created: int = Field(default=0, ge=0)
    aggregation_specs_skipped: int = Field(default=0, ge=0)
    aggregate_runs_created: int = Field(default=0, ge=0)
    aggregate_runs_skipped: int = Field(default=0, ge=0)

    # Dataset finalization (created in open/close)
    aggregated_datasets_created: int = Field(default=0, ge=0)
    aggregated_datasets_skipped: int = Field(default=0, ge=0)

    # Per-vote entities (streamed during write_one)
    aggregated_votes_created: int = Field(default=0, ge=0)
    aggregated_votes_skipped: int = Field(default=0, ge=0)
    aggregation_votes_created: int = Field(default=0, ge=0)
    aggregation_votes_skipped: int = Field(default=0, ge=0)
    aggregated_dataset_votes_created: int = Field(default=0, ge=0)

    def add_aggregation_specs(self, created: int = 0, skipped: int = 0) -> None:
        """Increment aggregation spec counts."""
        self.aggregation_specs_created += created
        self.aggregation_specs_skipped += skipped

    def add_aggregate_runs(self, created: int = 0, skipped: int = 0) -> None:
        """Increment aggregate run counts."""
        self.aggregate_runs_created += created
        self.aggregate_runs_skipped += skipped

    def add_aggregated_datasets(self, created: int = 0, skipped: int = 0) -> None:
        """Increment aggregated dataset counts."""
        self.aggregated_datasets_created += created
        self.aggregated_datasets_skipped += skipped

    def add_aggregated_votes(self, created: int = 0, skipped: int = 0) -> None:
        """Increment aggregated vote counts."""
        self.aggregated_votes_created += created
        self.aggregated_votes_skipped += skipped

    def add_aggregation_votes(self, created: int = 0, skipped: int = 0) -> None:
        """Increment aggregation vote counts."""
        self.aggregation_votes_created += created
        self.aggregation_votes_skipped += skipped

    def add_aggregated_dataset_votes(self, created: int = 0) -> None:
        """Increment aggregated dataset vote junction counts."""
        self.aggregated_dataset_votes_created += created

    @property
    def total_created(self) -> int:
        """Total entities created across all types."""
        return (
            self.aggregation_specs_created
            + self.aggregate_runs_created
            + self.aggregated_datasets_created
            + self.aggregated_votes_created
            + self.aggregation_votes_created
            + self.aggregated_dataset_votes_created
        )

    @property
    def total_skipped(self) -> int:
        """Total entities skipped across all types."""
        return (
            self.aggregation_specs_skipped
            + self.aggregate_runs_skipped
            + self.aggregated_datasets_skipped
            + self.aggregated_votes_skipped
            + self.aggregation_votes_skipped
        )
