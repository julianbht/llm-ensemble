"""InferRunInfo schema - runtime metadata for infer CLI runs.

Contains only run metadata (git info, timestamps, notes, run_type).
Configuration and execution context are separated into:
- InferRunConfig: Model/adapter/retry configuration
- InferRunContext: CLI args (input_run_name, start_idx, end_idx, io_name)
- InferRunOutput: Judgements and aggregate metrics produced

Separation of concerns:
- InferRunInfo: Run metadata (git SHA, timestamps, run_type, notes)
- InferRunConfig: Configuration used to produce judgements
- InferRunContext: Execution context (input source, sample range)
- InferRunOutput: Actual judgements and metrics produced
"""

from __future__ import annotations
from typing import Optional
from pydantic import ConfigDict, Field

from llm_ensemble.libs.runtime.run_info import RunInfo, RunType
from llm_ensemble.libs.runtime.run_name import generate_run_name


class InferRunInfo(RunInfo):
    """Runtime metadata for infer CLI runs.

    Contains only run metadata inherited from RunInfo:
    - Run identification (id, run_name, cli_name)
    - Run type (official vs test)
    - User context (notes)
    - Git metadata (commit SHA, branch, clean status)
    - Timestamps (start_time, end_time via RunInfo)

    Configuration and execution context are separated into InferRunConfig and
    InferRunContext respectively. This keeps concerns cleanly separated.
    """

    # Override cli_name from base RunInfo to automatically set it to "infer"
    cli_name: str = Field(
        default="infer",
        description="Name of the CLI that generated this run (always 'infer' for InferRunInfo)"
    )

    model_config = ConfigDict(frozen=True)

    @classmethod
    def create(
        cls,
        run_name: Optional[str] = None,
        name_hints: Optional[list[str]] = None,
        official: bool = False,
        notes: Optional[str] = None,
    ) -> "InferRunInfo":
        """Factory method to create InferRunInfo with automatic run_name generation.

        Args:
            run_name: Custom run ID (auto-generates from name_hints if not provided)
            name_hints: List of name hints for run ID generation (e.g., [model_hint, prompt_hint])
            official: Mark as official run (saved to official/ subdirectory for git tracking)
            notes: Optional user-provided notes about this run

        Returns:
            InferRunInfo instance with all fields populated (including id and git_info via defaults)
        """
        if run_name is None:
            if name_hints is None:
                raise ValueError("Either run_name or name_hints must be provided")
            run_name = generate_run_name(name_hints)

        return cls(
            run_name=run_name,
            run_type=RunType.OFFICIAL if official else RunType.TEST,
            notes=notes,
        )
