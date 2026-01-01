"""Driven port for reading evaluation inputs.

Driven Port Interface (Hexagonal Architecture)

This port abstracts input reading from infrastructure details.
The application depends on this abstraction, not concrete implementations.

Adapters implement this port to provide different input sources
(database, JSON files, etc.).
"""

from __future__ import annotations
from abc import ABC, abstractmethod

from llm_ensemble.evaluate.domain.entities.evaluation_data import EvaluationData


class ForInput(ABC):
    """Driven port for reading normalized evaluation data.

    The application depends on this abstraction.
    IO adapters implement this interface.

    Responsibilities:
    - Read judgements from infer or aggregate runs
    - Normalize to EvaluationData entity (ground_truth vs predictions)
    - Validate data via domain builder
    """

    @abstractmethod
    def read(self, input_run_name: str) -> EvaluationData:
        """Read and normalize evaluation data from input run.

        Args:
            input_run_name: Run name to read from (infer or aggregate run)

        Returns:
            EvaluationData entity with validated ground truth and predictions

        Raises:
            FileNotFoundError: If input run not found
            ValueError: If input format invalid or violates business rules
        """
        pass
