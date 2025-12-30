"""Port interface for aggregated dataset writers.

Defines the abstract contract for writing AggregateRun records to storage.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

from llm_ensemble.aggregate.domain.entities.aggregate_run import AggregateRun
from llm_ensemble.aggregate.domain.entities.write_summary import WriteSummary


class ForOutput(ABC):
    """Abstract base class for writing AggregateRun records.

    Implementations can write to different formats (Database, JSON, etc.)
    while providing a consistent interface.

    Supports batch writing of entire aggregate run (config + dataset + metadata).
    """

    @property
    @abstractmethod
    def io_name(self) -> str:
        """Get I/O adapter name for this output port.

        Returns:
            I/O adapter name (e.g., 'db_to_db', 'json')
        """
        pass

    @abstractmethod
    def write(self, aggregate_run: AggregateRun) -> WriteSummary:
        """Write entire aggregate run in one batch.

        Batch persistence pattern (like ingest):
        - Write all entities in a single transaction
        - Return summary of what was created/skipped

        Args:
            aggregate_run: The complete aggregate run entity with config, dataset, and metadata

        Returns:
            WriteSummary tracking what entities were created/skipped

        Raises:
            IOError: If write operation fails
        """
        pass
