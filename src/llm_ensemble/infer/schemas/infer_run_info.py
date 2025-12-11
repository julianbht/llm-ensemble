"""InferRunInfo schema - runtime context for infer CLI runs.

Contains only CLI parameters and run metadata. Configuration objects belong on
the JudgedDataset entity, not here.

Separation of concerns:
- InferRunInfo: CLI parameters + run metadata (what was requested)
- JudgedDataset: Model + adapter configs (what was used to produce judgements)
- InferRunORM: Links the two and tracks intent vs. actual result
"""

from __future__ import annotations
from typing import Optional
from pydantic import ConfigDict, Field

from llm_ensemble.libs.runtime.run_info import RunInfo, RunType
from llm_ensemble.libs.runtime.run_name import generate_run_name


class InferRunInfo(RunInfo):
    """Runtime context for infer CLI runs.

    Contains:
    1. Run metadata (inherited from RunInfo): id, run_name, run_type, notes, git info
    2. CLI parameters: input source and index range

    Configuration objects (ModelConfig, AdapterConfig) belong on JudgedDataset,
    not here. This keeps InferRunInfo focused on "what was requested" while
    JudgedDataset tracks "what was actually used".
    """

    # Override cli_name from base RunInfo to automatically set it to "infer"
    cli_name: str = Field(
        default="infer",
        description="Name of the CLI that generated this run (always 'infer' for InferRunInfo)"
    )

    # CLI parameters - input source
    input_run_name: str = Field(
        ...,
        description="Ingest run name to read samples from (e.g., 'my_ingest_run')"
    )

    # CLI parameters - index range (optional, from --start-idx and --end-idx flags)
    start_idx: Optional[int] = Field(
        default=None,
        description="Start index into NormalizedDataset.samples (0-indexed, inclusive, None = start from beginning)"
    )

    end_idx: Optional[int] = Field(
        default=None,
        description="End index into NormalizedDataset.samples (exclusive, None = process until end)"
    )

    model_config = ConfigDict(frozen=True)

    @classmethod
    def create(
        cls,
        name_hints: list[str],
        input_run_name: str,
        run_name: Optional[str] = None,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
        official: bool = False,
        notes: Optional[str] = None,
    ) -> "InferRunInfo":
        """Factory method to create InferRunInfo with automatic run_name generation.

        Args:
            name_hints: List of name hints for run ID generation (e.g., [model_hint, prompt_hint, io_hint])
            input_run_name: Ingest run name to read samples from
            run_name: Custom run ID (auto-generates from name_hints if not provided)
            start_idx: Start index into NormalizedDataset (None = start from beginning)
            end_idx: End index into NormalizedDataset (None = process until end)
            official: Mark as official run (saved to official/ subdirectory for git tracking)
            notes: Optional user-provided notes about this run

        Returns:
            InferRunInfo instance with all fields populated (including id and git_info via defaults)
        """
        if run_name is None:
            run_name = generate_run_name(name_hints)

        return cls(
            run_name=run_name,
            input_run_name=input_run_name,
            start_idx=start_idx,
            end_idx=end_idx,
            run_type=RunType.OFFICIAL if official else RunType.TEST,
            notes=notes,
        )
