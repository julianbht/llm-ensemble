"""Driving port for running evaluation pipeline.

Driving Port Interface (Hexagonal Architecture)

This is the application's driving port - the contract that driving adapters
(CLI, Web API, etc.) use to interact with the evaluation use case.

Driving adapters call this interface to execute the evaluation backend.
The application layer implements this port.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional


class ForRunningEvaluation(ABC):
    """Driving port for evaluation pipeline execution.

    Driving adapters (CLI, Web API) depend on this abstraction.
    The application use case implements this interface.

    This defines the application's public API - what driving adapters can request.
    """

    @abstractmethod
    def run_evaluation(
        self,
        input_run_name: str,
        official: bool,
        notes: Optional[str],
    ) -> None:
        """Execute the evaluation pipeline.

        Args:
            input_run_name: Run name to evaluate (infer or aggregate run)
            official: Mark as official run
            notes: Optional notes about this evaluation

        Returns:
            None (outputs written to disk)

        Raises:
            Exception: If any step in the pipeline fails
        """
        pass
