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
    Used as metadata in run summaries - NOT for logging (adapters log directly).
    """

    model_config = ConfigDict(validate_assignment=True)

    aggregation_strategies_created: int = Field(default=0, ge=0, description="Number of aggregation strategies created")
    aggregation_strategies_skipped: int = Field(default=0, ge=0, description="Number of aggregation strategies skipped (already existed)")
    configs_created: int = Field(default=0, ge=0, description="Number of aggregate run configs created")
    configs_skipped: int = Field(default=0, ge=0, description="Number of aggregate run configs skipped (already existed)")
    datasets_created: int = Field(default=0, ge=0, description="Number of aggregated datasets created")
    datasets_skipped: int = Field(default=0, ge=0, description="Number of aggregated datasets skipped (already existed)")
    aggregated_votes_created: int = Field(default=0, ge=0, description="Number of aggregated votes created")
    aggregated_votes_skipped: int = Field(default=0, ge=0, description="Number of aggregated votes skipped (already existed)")
    aggregation_votes_created: int = Field(default=0, ge=0, description="Number of aggregation vote junctions created")
    aggregation_votes_skipped: int = Field(default=0, ge=0, description="Number of aggregation vote junctions skipped (already existed)")
    aggregated_dataset_votes_created: int = Field(default=0, ge=0, description="Number of aggregated dataset vote junctions created")
    aggregated_dataset_votes_skipped: int = Field(default=0, ge=0, description="Number of aggregated dataset vote junctions skipped (already existed)")
    runs_created: int = Field(default=0, ge=0, description="Number of aggregate runs created")
    runs_skipped: int = Field(default=0, ge=0, description="Number of aggregate runs skipped (already existed)")

    def add_aggregation_strategies(self, created: int = 0, skipped: int = 0) -> None:
        """Increment aggregation strategy counts."""
        self.aggregation_strategies_created += created
        self.aggregation_strategies_skipped += skipped

    def add_configs(self, created: int = 0, skipped: int = 0) -> None:
        """Increment config counts."""
        self.configs_created += created
        self.configs_skipped += skipped

    def add_aggregated_datasets(self, created: int = 0, skipped: int = 0) -> None:
        """Increment aggregated dataset counts."""
        self.datasets_created += created
        self.datasets_skipped += skipped

    def add_aggregated_votes(self, created: int = 0, skipped: int = 0) -> None:
        """Increment aggregated vote counts."""
        self.aggregated_votes_created += created
        self.aggregated_votes_skipped += skipped

    def add_aggregation_votes(self, created: int = 0, skipped: int = 0) -> None:
        """Increment aggregation vote junction counts."""
        self.aggregation_votes_created += created
        self.aggregation_votes_skipped += skipped

    def add_aggregated_dataset_votes(self, created: int = 0, skipped: int = 0) -> None:
        """Increment aggregated dataset vote junction counts."""
        self.aggregated_dataset_votes_created += created
        self.aggregated_dataset_votes_skipped += skipped

    def add_aggregate_runs(self, created: int = 0, skipped: int = 0) -> None:
        """Increment aggregate run counts."""
        self.runs_created += created
        self.runs_skipped += skipped

    @property
    def total_created(self) -> int:
        """Total entities created across all types."""
        return (
            self.aggregation_strategies_created
            + self.configs_created
            + self.datasets_created
            + self.aggregated_votes_created
            + self.aggregation_votes_created
            + self.aggregated_dataset_votes_created
            + self.runs_created
        )

    @property
    def total_skipped(self) -> int:
        """Total entities skipped across all types."""
        return (
            self.aggregation_strategies_skipped
            + self.configs_skipped
            + self.datasets_skipped
            + self.aggregated_votes_skipped
            + self.aggregation_votes_skipped
            + self.aggregated_dataset_votes_skipped
            + self.runs_skipped
        )
