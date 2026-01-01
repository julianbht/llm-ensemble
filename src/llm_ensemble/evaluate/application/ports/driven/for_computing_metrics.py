"""Driven port for computing evaluation metrics.

Driven Port Interface (Hexagonal Architecture)

This port abstracts metric computation from infrastructure details.
The application depends on this abstraction, not concrete implementations.

Metric adapters implement this port to provide different metrics
(Cohen's Kappa, Kendall's Tau, etc.).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional

from llm_ensemble.libs.schemas.relevance_score import RelevanceScore
from llm_ensemble.evaluate.domain.entities.metric_result import MetricResult


class ForComputingMetrics(ABC):
    """Driven port for computing evaluation metrics.

    The application depends on this abstraction.
    Metric adapters implement this interface.

    Each metric adapter computes a specific metric (Cohen's Kappa, etc.)
    and returns a standardized MetricResult entity.
    """

    @abstractmethod
    def compute(
        self,
        ground_truth: list[RelevanceScore],
        predictions: list[Optional[RelevanceScore]]
    ) -> MetricResult:
        """Compute metric from ground truth and predictions.

        Args:
            ground_truth: List of ground truth relevance labels
            predictions: List of predicted relevance labels (None if parse failed)

        Returns:
            MetricResult entity with metric value and metadata

        Raises:
            ValueError: If inputs invalid or incompatible
        """
        pass
