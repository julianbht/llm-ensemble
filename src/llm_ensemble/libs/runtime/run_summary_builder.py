"""RunSummary builder using the Builder pattern.

Provides a RunSummaryBuilder for constructing CLI-specific run summaries step-by-step.
The builder separates summary construction from the final Pydantic representation.
Domain services can add metrics incrementally during execution before finalizing.

This replaces the old ManifestBuilder, splitting concerns:
- RunInfo: Immutable runtime context known before run starts
- RunSummary: Aggregate metrics computed after run completes
"""

from __future__ import annotations
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from llm_ensemble.libs.runtime.run_info import RunInfo


class RunSummaryBuilder:
    """Builder for constructing CLI-specific run summaries step-by-step.

    This implements the Builder pattern, allowing domain services to:
    1. Set start_time when processing begins
    2. Add aggregate metrics incrementally as they're computed
    3. Finalize to create the immutable Pydantic RunSummary at the end

    The builder collects timing and aggregate statistics during execution.
    RunInfo is persisted separately to avoid duplication in the summary.

    Example:
        >>> # Domain service creates builder
        >>> builder = RunSummaryBuilder()
        >>>
        >>> # Domain service sets start time when processing begins
        >>> builder.set_start_time()
        >>>
        >>> # Domain service adds metrics as they're computed
        >>> builder.add("judgement_count", 100)
        >>> builder.add("error_count", 5)
        >>> builder.add("total_latency_ms", 45000.0)
        >>> builder.add("avg_latency_ms", 450.0)
        >>>
        >>> # Domain service finalizes (sets end_time and creates immutable summary)
        >>> summary = builder.finalize(InferRunSummary)
    """

    def __init__(self):
        """Initialize run summary builder for collecting runtime metrics."""
        # Initialize fields with timing placeholders
        self._fields: dict[str, Any] = {
            "start_time": None,  # Set by domain service when processing begins
            "end_time": None,    # Set during finalize()
        }

    def set_start_time(self, start_time: datetime | None = None) -> "RunSummaryBuilder":
        """Set the start time (when domain service begins processing).

        Args:
            start_time: Start timestamp (defaults to now if not provided)

        Returns:
            Self for method chaining (Fluent Builder pattern)

        Example:
            >>> builder.set_start_time()  # Uses datetime.now()
            >>> builder.set_start_time(custom_start)  # Uses provided timestamp
        """
        self._fields["start_time"] = start_time or datetime.now()
        return self

    def add(self, key: str, value: Any) -> "RunSummaryBuilder":
        """Add an aggregate metric or statistic to the summary.

        Args:
            key: Field name (e.g., "judgement_count", "avg_latency_ms")
            value: Field value

        Returns:
            Self for method chaining (Fluent Builder pattern)

        Example:
            >>> builder.add("judgement_count", 100).add("error_count", 5)
        """
        self._fields[key] = value
        return self

    def finalize(self, summary_class: type[BaseModel]) -> BaseModel:
        """Finalize the summary by setting end_time and creating the Pydantic object.

        Args:
            summary_class: The Pydantic model class to instantiate (e.g., InferRunSummary)

        Returns:
            Immutable Pydantic run summary object

        Example:
            >>> summary = builder.finalize(InferRunSummary)
        """
        # Set end_time to mark completion
        self._fields["end_time"] = datetime.now()

        # Create and validate Pydantic summary
        return summary_class(**self._fields)


def write_standalone_summary(summary: BaseModel, run_dir: Any) -> Any:
    """Write a standalone summary.json for quick inspection of runtime metrics.

    This writes a simplified metrics-only summary for human convenience.
    The summary contains only runtime statistics (timing, counts, errors, etc.)
    without configuration duplication. Configuration is persisted separately
    (e.g., infer_run_info.json).

    This function is provided for:
    - Quick inspection of run performance without loading entity files
    - Debugging and exploration
    - At-a-glance view of runtime metrics

    Args:
        summary: Pydantic RunSummary object (base or CLI-specific subclass)
        run_dir: Run directory path

    Returns:
        Path to the written summary file

    Example:
        >>> # Write simplified metrics-only summary
        >>> summary_path = write_standalone_summary(summary, run_dir)
    """
    from pathlib import Path

    # Ensure run directory exists
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write summary as JSON (using "summary.json" instead of "manifest.json")
    summary_path = run_dir / "summary.json"
    summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")

    return summary_path
