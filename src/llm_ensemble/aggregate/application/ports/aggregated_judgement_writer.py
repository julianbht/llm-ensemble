"""Port interface for aggregated dataset writers.

Defines the abstract contract for writing AggregatedDataset records to storage.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path

from llm_ensemble.aggregate.domain.entities.aggregated_dataset import AggregatedDataset
from llm_ensemble.aggregate.domain.entities.aggregate_run_info import AggregateRunInfo
from llm_ensemble.aggregate.domain.entities.write_summary import WriteSummary


class AggregatedJudgementWriter(ABC):
    """Abstract base class for writing AggregatedDataset records.

    Implementations can write to different formats (Database, JSON, etc.)
    while providing a consistent interface.

    Supports batch writing of entire aggregated dataset.
    """

    @abstractmethod
    def write(
        self,
        run_dir: Path,
        run_info: AggregateRunInfo,
        aggregated_dataset: AggregatedDataset,
    ) -> WriteSummary:
        """Write entire aggregated dataset in one batch.

        Args:
            run_dir: Run directory where output should be written
            run_info: Aggregate run context (config, metadata)
            aggregated_dataset: The aggregated dataset to write (with all votes)

        Returns:
            WriteSummary tracking what entities were created/skipped

        Raises:
            IOError: If write operation fails
        """
        pass
