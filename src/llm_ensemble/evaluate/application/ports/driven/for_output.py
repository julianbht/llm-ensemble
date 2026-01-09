"""Port interface for evaluation output writers.

Defines the abstract contract for writing EvaluateRun records to storage.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

from llm_ensemble.evaluate.domain.entities.evaluate_run import EvaluateRun


class ForOutput(ABC):
    """Abstract base class for writing EvaluateRun records.

    Implementations can write to different formats (Database, JSON, etc.)
    while providing a consistent interface.

    Supports batch writing of entire evaluate run (config + metrics + metadata).
    """

    @property
    @abstractmethod
    def io_name(self) -> str:
        """Get I/O adapter name for this output port.

        Returns:
            I/O adapter name (e.g., 'json', 'dummy', 'db_evaluate')
        """
        pass

    @abstractmethod
    def write(self, evaluate_run: EvaluateRun) -> None:
        """Write entire evaluate run in one batch.

        Args:
            evaluate_run: The complete evaluate run entity with config, metrics, and metadata

        Raises:
            IOError: If write operation fails
        """
        pass
