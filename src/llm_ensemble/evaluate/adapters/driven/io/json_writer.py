"""JSON writer adapter for persisting evaluation runs.

Driven Adapter - I/O (Hexagonal Architecture)

Implements ForOutput port by writing EvaluateRun entity to JSON file.
Uses Pydantic's built-in serialization for clean, structured output.
"""

from __future__ import annotations
from pathlib import Path

from llm_ensemble.evaluate.application.ports.driven.for_output import ForOutput
from llm_ensemble.evaluate.domain.entities.evaluate_run import EvaluateRun
from llm_ensemble.libs.logging.structlog_logger import get_logger


class JSONWriter(ForOutput):
    """JSON output adapter for evaluation runs.

    Implements ForOutput port by writing complete EvaluateRun entity to JSON.
    Uses Pydantic's model_dump_json() for automatic serialization.
    """

    def __init__(self, io_name: str, run_dir: Path):
        """Initialize JSON writer.

        Args:
            io_name: I/O configuration name
            run_dir: Run directory path (where evaluate_run.json will be written)
        """
        self._io_name = io_name
        self.run_dir = run_dir
        self.logger = get_logger(component=__name__)

    @property
    def io_name(self) -> str:
        """Get I/O adapter name."""
        return self._io_name

    def write(self, evaluate_run: EvaluateRun) -> None:
        """Write complete EvaluateRun entity to JSON file.

        Output: <run_dir>/evaluate_run.json

        Args:
            evaluate_run: Complete evaluate run entity

        Raises:
            IOError: If writing fails
        """
        try:
            output_path = self.run_dir / "evaluate_run.json"

            # Use Pydantic's built-in JSON serialization
            json_str = evaluate_run.model_dump_json(indent=2)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(json_str)

            self.logger.info(
                "evaluate.json_writer.write_complete",
                path=str(output_path),
                metric_count=len(evaluate_run.metric_results)
            )

        except Exception as e:
            raise IOError(f"Failed to write evaluate run to JSON: {e}") from e
