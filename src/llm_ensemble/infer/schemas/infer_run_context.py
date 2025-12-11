"""InferRunContext - execution context for an infer run.

This entity captures CLI arguments that control execution behavior but are not
part of the configuration itself:
- Input source (which ingest run to read from)
- Index range (which samples to process)
- I/O configuration (format for reading/writing)

This is separate from:
- InferRunInfo: Git info, timestamps, run metadata
- InferRunConfig: Model/adapter/retry configuration
- InferRunOutput: The actual judgements and metrics produced

Responsibilities:
- Capture CLI args that affect execution (not configuration)
- Provide context for run execution (input source, sample range)
- Part of the overall run configuration but separate concern
"""

from __future__ import annotations
from typing import Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, ConfigDict, Field


class InferRunContext(BaseModel):
    """Execution context for an infer run.

    Contains CLI arguments that control execution behavior:
    - Input source: which ingest run to read samples from
    - Index range: which samples to process (start_idx, end_idx)
    - I/O format: how to read/write data (io_name)

    This represents "how the run was executed" (input source, sample selection),
    separate from "what configuration was used" (InferRunConfig) and
    "what was produced" (InferRunOutput).
    """

    id: UUID = Field(
        default_factory=uuid4,
        description="Random UUID for this context"
    )

    input_run_name: str = Field(
        ...,
        description="Ingest run name to read samples from (e.g., 'my_ingest_run')"
    )

    io_name: str = Field(
        ...,
        description="I/O format name (e.g., 'db_to_json', 'db_to_db')"
    )

    start_idx: Optional[int] = Field(
        default=None,
        description="Start index into NormalizedDataset.samples (0-indexed, inclusive, None = start from beginning)"
    )

    end_idx: Optional[int] = Field(
        default=None,
        description="End index into NormalizedDataset.samples (exclusive, None = process until end)"
    )

    model_config = ConfigDict(frozen=True)
