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
    1. Initialize with RunInfo (immutable runtime context from orchestrator)
    2. Set start_time when processing begins
    3. Add aggregate metrics incrementally as they're computed
    4. Finalize to create the immutable Pydantic RunSummary at the end

    The builder starts with a RunInfo object (created by orchestrator) which
    contains all immutable context. The builder then collects timing and
    aggregate statistics during execution.

    Example:
        >>> # Orchestrator creates RunInfo
        >>> run_info = InferRunInfo(
        ...     run_id="20250128_143022_gpt-oss-20b",
        ...     cli_name="infer",
        ...     run_type="test",
        ...     git_sha="abc123",
        ...     git_clean=True,
        ...     git_branch="master",
        ...     model_config_name="gpt-oss-20b",
        ...     model_cfg=model_config,
        ...     ...
        ... )
        >>>
        >>> # Domain service creates builder with RunInfo
        >>> builder = RunSummaryBuilder(run_info)
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

    def __init__(self, run_info: RunInfo):
        """Initialize run summary builder with immutable runtime context.

        Args:
            run_info: Immutable runtime context (created by orchestrator before run starts)
        """
        self.run_info = run_info

        # Initialize fields with RunInfo and timing placeholders
        self._fields: dict[str, Any] = {
            "run_info": run_info,
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
    """Write a standalone summary.json for human convenience (optional).

    NOTE: This is for quick inspection only. The source of truth for summaries
    is embedded in the domain data (e.g., LLMJudgements contain run_info).
    I/O adapters are responsible for persisting summaries with domain data.

    This function is provided as a convenience for:
    - Quick inspection of run metadata without loading all samples
    - Debugging and exploration
    - Compatibility with tools expecting summary.json files

    Args:
        summary: Pydantic RunSummary object (base or CLI-specific subclass)
        run_dir: Run directory path

    Returns:
        Path to the written summary file

    Example:
        >>> # Optionally write standalone summary for convenience
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
