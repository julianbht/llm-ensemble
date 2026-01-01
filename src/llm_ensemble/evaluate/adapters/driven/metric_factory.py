"""Factory for creating metric adapter instances.

Explicit instantiation of metric adapters with metric-specific constructors.
Each adapter defines its own constructor signature and configuration needs.

To add a new metric:
1. Create adapter class that extends ForComputingMetrics
2. Import it here
3. Add explicit instantiation case in create() method
"""

from __future__ import annotations

from llm_ensemble.evaluate.application.ports.driven.for_computing_metrics import ForComputingMetrics
from llm_ensemble.evaluate.adapters.driven.metrics.dummy_metric import DummyMetricAdapter


AVAILABLE_METRICS = ["dummy"]


class MetricAdapterFactory:
    """Factory for creating metric adapter instances."""

    @staticmethod
    def create(metric_name: str) -> ForComputingMetrics:
        """Build and return a metric adapter instance.

        Args:
            metric_name: Name of the metric (e.g., 'cohens_kappa', 'kendalls_tau')

        Returns:
            Instantiated metric adapter

        Raises:
            ValueError: If metric not found
        """
        if metric_name == "dummy":
            return DummyMetricAdapter()
        else:
            available = ", ".join(sorted(AVAILABLE_METRICS))
            raise ValueError(
                f"Metric '{metric_name}' not found. "
                f"Available: {available}"
            )

    @staticmethod
    def list_available() -> list[str]:
        """List all available metric names."""
        return sorted(AVAILABLE_METRICS)

    @staticmethod
    def has_metric(metric_name: str) -> bool:
        """Check if metric is available."""
        return metric_name in AVAILABLE_METRICS
