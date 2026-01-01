"""Driven port for reading evaluation inputs.

Driven Port Interface (Hexagonal Architecture)

This port abstracts input reading from infrastructure details.
The application depends on this abstraction, not concrete implementations.

Adapters implement this port to provide different input sources
(database, JSON files, etc.).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class ForInput(ABC):
    """Driven port for reading normalized evaluation data.

    The application depends on this abstraction.
    IO adapters implement this interface.

    Responsibilities:
    - Read judgements from infer or aggregate runs
    - Normalize to a common format (ground_truth vs predicted)
    - Extract relevant metadata
    """

    @abstractmethod
    def read(self, input_run_name: str) -> Any:
        """Read and normalize evaluation data from input run.

        Args:
            input_run_name: Run name to read from (infer or aggregate run)

        Returns:
            Normalized evaluation data (to be defined)

        Raises:
            FileNotFoundError: If input run not found
            ValueError: If input format invalid
        """
        pass
