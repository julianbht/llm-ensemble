"""Dummy metric adapter for testing infrastructure.

Driven Adapter - Metrics (Hexagonal Architecture)

Implements ForComputingMetrics port with a placeholder metric.
This adapter demonstrates the metric adapter pattern and can be
replaced with real metrics (Cohen's Kappa, Kendall's Tau, etc.).
"""

from __future__ import annotations
from typing import Any

from llm_ensemble.evaluate.application.ports.driven.for_computing_metrics import ForComputingMetrics


class DummyMetricAdapter(ForComputingMetrics):
    """Dummy metric adapter that returns placeholder values.

    Implements ForComputingMetrics port for testing infrastructure.
    Replace with real metric implementations.
    """

    def compute(self, ground_truth: list[Any], predictions: list[Any]) -> Any:
        """Compute dummy metric (returns placeholder value).

        Args:
            ground_truth: List of ground truth labels
            predictions: List of predicted labels

        Returns:
            Placeholder metric result (dict with dummy values)

        Raises:
            ValueError: If inputs have different lengths
        """
        if len(ground_truth) != len(predictions):
            raise ValueError(
                f"Ground truth and predictions must have same length "
                f"(got {len(ground_truth)} vs {len(predictions)})"
            )

        # Dummy metric: return placeholder result
        return {
            "name": "dummy_metric",
            "type": "scalar",
            "value": 0.85,
            "metadata": {
                "sample_size": len(ground_truth),
                "description": "Placeholder metric for testing infrastructure",
            },
        }
