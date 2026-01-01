"""Driven port for computing evaluation metrics.

Driven Port Interface (Hexagonal Architecture)

This port abstracts metric computation from infrastructure details.
The application depends on this abstraction, not concrete implementations.

Metric adapters implement this port to provide different metrics
(Cohen's Kappa, Kendall's Tau, etc.).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class ForComputingMetrics(ABC):
    """Driven port for computing evaluation metrics.

    The application depends on this abstraction.
    Metric adapters implement this interface.

    Each metric adapter computes a specific metric (Cohen's Kappa, etc.)
    and returns a standardized result structure.
    """

    @abstractmethod
    def compute(self, ground_truth: list[Any], predictions: list[Any]) -> Any:
        """Compute metric from ground truth and predictions.

        Args:
            ground_truth: List of ground truth labels
            predictions: List of predicted labels

        Returns:
            Metric result (to be defined as MetricResult)

        Raises:
            ValueError: If inputs invalid or incompatible
        """
        pass
