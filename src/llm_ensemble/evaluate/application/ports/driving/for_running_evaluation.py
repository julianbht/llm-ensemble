"""Driving port for running evaluation pipeline.

Driving Port (Primary/Driver Port) - Application API

This is the interface that the application OFFERS to driving adapters.
Driving adapters (CLI, Web API, Test harness, etc.) call this interface.

Defined BY the application, implemented BY the application (EvaluationApplication).
Called BY driving adapters.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional


class ForRunningEvaluation(ABC):
    """
    Driving port for evaluation pipeline execution.
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
