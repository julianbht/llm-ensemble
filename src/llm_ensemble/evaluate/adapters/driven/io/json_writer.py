"""JSON writer adapter for persisting evaluation metrics.

Driven Adapter - I/O (Hexagonal Architecture)

Implements ForOutput port by writing metrics to JSON file.
This adapter provides structured, machine-readable evaluation results
suitable for further analysis, visualization, or downstream processing.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from llm_ensemble.evaluate.application.ports.driven.for_output import ForOutput
from llm_ensemble.evaluate.domain.entities.metric_result import MetricResult
from llm_ensemble.libs.logging.structlog_logger import get_logger


class JSONWriter(ForOutput):
    """JSON output adapter for evaluation metrics.

    Implements ForOutput port by writing metrics and metadata to JSON file.
    Output file is written to: <run_dir>/metrics.json

    Output format:
    {
        "metadata": {
            "run_name": "...",
            "input_run_name": "...",
            "input_run_type": "...",
            "sample_count": 100,
            "official": false,
            "notes": null
        },
        "metrics": [
            {
                "name": "cohens_kappa",
                "value": 0.75,
                "sample_size": 100,
                "interpretation": "substantial",
                "description": "Cohen's Kappa coefficient..."
            }
        ]
    }
    """

    def __init__(self, io_name: str, run_dir: Path):
        """Initialize JSON writer.

        Args:
            io_name: I/O configuration name
            run_dir: Run directory path (where metrics.json will be written)
        """
        self.io_name = io_name
        self.run_dir = run_dir
        self.logger = get_logger(component=__name__)

    def write(self, metric_results: list[MetricResult], run_metadata: dict[str, Any]) -> None:
        """Write evaluation metrics to JSON file.

        Writes metrics and metadata to <run_dir>/metrics.json.
        The output file contains both metric results and run context.

        Args:
            metric_results: List of computed metric results
            run_metadata: Metadata about the evaluation run

        Raises:
            IOError: If writing fails
        """
        try:
            # Build output structure
            output_data = {
                "metadata": run_metadata,
                "metrics": [self._metric_to_dict(m) for m in metric_results]
            }

            # Write to file
            output_path = self.run_dir / "metrics.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            self.logger.info(
                "evaluate.json_writer.write_complete",
                path=str(output_path),
                metric_count=len(metric_results)
            )

        except Exception as e:
            raise IOError(f"Failed to write metrics to JSON: {e}") from e

    def _metric_to_dict(self, metric: MetricResult) -> dict[str, Any]:
        """Convert MetricResult to dictionary.

        Args:
            metric: MetricResult entity

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            "name": metric.name,
            "value": metric.value,
            "sample_size": metric.sample_size,
            "interpretation": metric.interpretation,
            "description": metric.description,
        }
