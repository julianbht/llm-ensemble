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

class RunSummaryBuilder:
    """Builder for constructing CLI-specific run summaries step-by-step.

    This implements the Builder pattern, allowing domain services to:
    1. Set start_time when processing begins
    2. Add aggregate metrics incrementally as they're computed
    3. Finalize to create the immutable Pydantic RunSummary at the end

    The builder collects timing and aggregate statistics during execution.
    """

    def __init__(self):
        """Initialize run summary builder for collecting runtime metrics."""
        # Initialize fields with timing placeholders
        self._fields: dict[str, Any] = {
            "start_time": None,  # Set by domain service when processing begins
            "end_time": None,    # Set during finalize()
        }

    def set_start_time(self, start_time: datetime | None = None) -> "RunSummaryBuilder":
        self._fields["start_time"] = start_time or datetime.now()
        return self

    def add(self, key: str, value: Any) -> "RunSummaryBuilder":
        self._fields[key] = value
        return self

    def finalize(self, summary_class: type[BaseModel]) -> BaseModel:
        # Set end_time to mark completion
        self._fields["end_time"] = datetime.now()

        # Create and validate Pydantic summary
        return summary_class(**self._fields)


def write_standalone_summary(summary: BaseModel, run_dir: Any) -> Any:
    from pathlib import Path

    # Ensure run directory exists
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write summary as JSON (using "summary.json" instead of "manifest.json")
    summary_path = run_dir / "summary.json"
    summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")

    return summary_path
