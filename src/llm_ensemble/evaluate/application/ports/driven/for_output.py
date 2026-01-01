"""Driven port for writing evaluation outputs.

Driven Port Interface (Hexagonal Architecture)

This port abstracts output writing from infrastructure details.
The application depends on this abstraction, not concrete implementations.

Adapters implement this port to provide different output formats
(HTML reports, JSON, etc.).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class ForOutput(ABC):
    """Driven port for writing evaluation reports.

    The application depends on this abstraction.
    IO adapters implement this interface.

    Responsibilities:
    - Write evaluation report to disk
    - Format metric results appropriately
    """

    @abstractmethod
    def write(self, metric_results: list[Any], run_metadata: Any) -> None:
        """Write evaluation report.

        Args:
            metric_results: List of metric computation results
            run_metadata: Metadata about the evaluation run

        Raises:
            IOError: If writing fails
        """
        pass
