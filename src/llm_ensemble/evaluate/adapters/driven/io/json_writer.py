"""JSON writer adapter for persisting evaluation runs.

Driven Adapter - I/O (Hexagonal Architecture)

Implements ForOutput port by writing EvaluateRun to separate JSON files.
This adapter provides structured, machine-readable evaluation results
suitable for further analysis, visualization, or downstream processing.

Output structure:
- config.json - EvaluateRunConfig entity
- run_metadata.json - run metadata (timestamps, git info, notes)
- evaluated_dataset_info.json - info about the evaluated dataset
- metrics/{metric_name}.json - one file per metric result
"""

from __future__ import annotations
import json
from pathlib import Path

from llm_ensemble.evaluate.application.ports.driven.for_output import ForOutput
from llm_ensemble.evaluate.domain.entities.evaluate_run import EvaluateRun
from llm_ensemble.evaluate.domain.entities.metric_result import MetricResult
from llm_ensemble.libs.logging.structlog_logger import get_logger


class JSONWriter(ForOutput):
    """JSON output adapter for evaluation runs.

    Implements ForOutput port by writing EvaluateRun entity to separate JSON files.
    Each first-level entity is written to its own file for easier analysis and querying.
    """

    def __init__(self, io_name: str, run_dir: Path):
        """Initialize JSON writer.

        Args:
            io_name: I/O configuration name
            run_dir: Run directory path (where JSON files will be written)
        """
        self._io_name = io_name
        self.run_dir = run_dir
        self.logger = get_logger(component=__name__)

    @property
    def io_name(self) -> str:
        """Get I/O adapter name."""
        return self._io_name

    def write(self, evaluate_run: EvaluateRun) -> None:
        """Write evaluation run to separate JSON files.

        Creates:
        - config.json - EvaluateRunConfig entity
        - run_metadata.json - run metadata (timestamps, git info, notes)
        - evaluated_dataset_info.json - info about the evaluated dataset
        - metrics/{metric_name}.json - one file per metric result

        Args:
            evaluate_run: Complete evaluate run entity

        Raises:
            IOError: If writing fails
        """
        try:
            # Write config
            self._write_config(evaluate_run)

            # Write run metadata
            self._write_run_metadata(evaluate_run)

            # Write evaluated dataset info
            self._write_evaluated_dataset_info(evaluate_run)

            # Write metric results (one file per metric)
            self._write_metric_results(evaluate_run)

            self.logger.info(
                "evaluate.json_writer.write_complete",
                run_dir=str(self.run_dir),
                metric_count=len(evaluate_run.metric_results)
            )

        except Exception as e:
            raise IOError(f"Failed to write evaluate run to JSON: {e}") from e

    def _write_config(self, evaluate_run: EvaluateRun) -> None:
        """Write EvaluateRunConfig to config.json."""
        config_data = {
            "id": str(evaluate_run.evaluate_run_config.id),
            "io_config_name": evaluate_run.evaluate_run_config.io_config_name,
            "input_run_name": evaluate_run.evaluate_run_config.input_run_name,
            "metric_names": evaluate_run.evaluate_run_config.metric_names,
        }

        output_path = self.run_dir / "config.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

    def _write_run_metadata(self, evaluate_run: EvaluateRun) -> None:
        """Write run metadata to run_metadata.json."""
        metadata = {
            "id": str(evaluate_run.id),
            "run_name": evaluate_run.run_name,
            "run_type": evaluate_run.run_type,
            "cli_name": evaluate_run.cli_name,
            "start_time": evaluate_run.start_time.isoformat(),
            "end_time": evaluate_run.end_time.isoformat(),
            "notes": evaluate_run.notes,
            "git_info": {
                "git_sha": evaluate_run.git_info.git_sha,
                "git_branch": evaluate_run.git_info.git_branch,
                "git_clean": evaluate_run.git_info.git_clean,
            }
        }

        output_path = self.run_dir / "run_metadata.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def _write_evaluated_dataset_info(self, evaluate_run: EvaluateRun) -> None:
        """Write evaluated dataset info to evaluated_dataset_info.json."""
        dataset_info = {
            "run_type": evaluate_run.evaluated_run_type,
            "sample_count": evaluate_run.evaluated_sample_count,
        }

        output_path = self.run_dir / "evaluated_dataset_info.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)

    def _write_metric_results(self, evaluate_run: EvaluateRun) -> None:
        """Write each metric result to metrics/{metric_name}.json."""
        metrics_dir = self.run_dir / "metrics"
        metrics_dir.mkdir(exist_ok=True)

        for metric in evaluate_run.metric_results:
            metric_data = {
                "name": metric.name,
                "value": metric.value,
                "sample_size": metric.sample_size,
                "interpretation": metric.interpretation,
                "description": metric.description,
            }

            output_path = metrics_dir / f"{metric.name}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(metric_data, f, indent=2, ensure_ascii=False)
