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

from llm_ensemble.libs.runtime.run_info import RunInfo


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
