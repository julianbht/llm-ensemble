"""InferRunInfo schema - extends base RunInfo with infer-specific configuration.

This contains all inference-specific configuration that is known before the run
starts and remains immutable throughout execution. By bundling this with the
base RunInfo, each LLMJudgement can carry complete provenance metadata without
waiting for the run to complete.
"""

from __future__ import annotations
import uuid
from typing import Optional
from uuid import UUID
from pydantic import ConfigDict, Field

from llm_ensemble.libs.runtime.run_info import RunInfo
from llm_ensemble.infer.schemas.model_config_schema import ModelConfig
from llm_ensemble.infer.schemas.retry_config_schema import RetryConfig


class InferRunInfo(RunInfo):
    """Runtime context for infer CLI runs.

    Extends the base RunInfo with infer-specific configuration metadata:
    - Which configs were used (model, prompt, I/O)
    - Full configuration objects for reproducibility
    - Input parameters (file path, limit)

    All fields in this class are immutable and known before processing begins,
    allowing LLMJudgement objects to embed complete provenance as soon as they
    are created, without waiting for aggregate statistics.

    This is separate from InferRunSummary which contains post-run metrics like
    judgement counts, timing statistics, and warnings summary.
    """

    # Random UUID
    id: UUID = Field(
        default_factory=uuid.uuid4,
        description="Random UUID for this run"
    )

    # Override cli_name from base RunInfo to automatically set it to "infer"
    cli_name: str = Field(
        default="infer",
        description="Name of the CLI that generated this run (always 'infer' for InferRunInfo)"
    )

    # Configuration names
    model_config_name: str = Field(
        ...,
        description="Name of the model config used (e.g., 'gpt-oss-20b')"
    )

    prompt_name: str = Field(
        ...,
        description="Name of the prompt used from registry (e.g., 'thomas-simple')"
    )

    parser_name: str = Field(
        ...,
        description="Name of the parser used from registry (e.g., 'thomas-simple')"
    )

    retry_config_name: str = Field(
        ...,
        description="Name of the retry config used (e.g., 'standard')"
    )

    io_name: str = Field(
        ...,
        description="Name of the I/O format used (e.g., 'db_to_json', 'db_to_db')"
    )

    # Full configuration objects (for reproducibility)
    # Note: Cannot use 'model_config' as field name (reserved by Pydantic v2)
    model_cfg: ModelConfig = Field(
        ...,
        description="Model configuration used for this run"
    )

    retry_config: RetryConfig = Field(
        ...,
        description="Retry configuration used for this run"
    )

    # Input parameters
    input_run_name: str = Field(
        ...,
        description="Ingest run name to read samples from (e.g., 'my_ingest_run')"
    )

    # Index range (optional, from CLI --start-idx and --end-idx flags)
    start_idx: Optional[int] = Field(
        default=None,
        description="Start index into NormalizedDataset.samples (0-indexed, inclusive, None = start from beginning)"
    )

    end_idx: Optional[int] = Field(
        default=None,
        description="End index into NormalizedDataset.samples (exclusive, None = process until end)"
    )

    model_config = ConfigDict(frozen=True)
