"""InferRunInfo schema - extends base RunInfo with infer-specific configuration.

This contains all inference-specific configuration that is known before the run
starts and remains immutable throughout execution. By bundling this with the
base RunInfo, each LLMJudgement can carry complete provenance metadata without
waiting for the run to complete.
"""

from __future__ import annotations
from typing import Optional
from uuid import UUID
from pydantic import ConfigDict, Field

from llm_ensemble.libs.runtime.run_info import RunInfo
from llm_ensemble.libs.schemas import IOConfig
from llm_ensemble.infer.schemas.model_config_schema import ModelConfig
from llm_ensemble.infer.schemas.prompt_config_schema import PromptConfig
from llm_ensemble.infer.schemas.retry_config_schema import RetryConfig
from llm_ensemble.libs.db import compute_infer_run_uuid


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
    
    The id field is a mandatory deterministic UUID computed from run_name.
    """

    # Deterministic UUID
    id: UUID = Field(
        ...,
        description="Deterministic UUID computed from run_name"
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

    prompt_config_name: str = Field(
        ...,
        description="Name of the prompt config used (e.g., 'thomas-et-al-prompt')"
    )

    retry_config_name: str = Field(
        ...,
        description="Name of the retry config used (e.g., 'standard')"
    )

    io_config_name: str = Field(
        ...,
        description="Name of the I/O config used (e.g., 'json')"
    )

    # Full configuration objects (for reproducibility)
    # Note: Cannot use 'model_config' as field name (reserved by Pydantic v2)
    model_cfg: ModelConfig = Field(
        ...,
        description="Model configuration used for this run"
    )

    prompt_config: PromptConfig = Field(
        ...,
        description="Prompt configuration used for this run"
    )

    retry_config: RetryConfig = Field(
        ...,
        description="Retry configuration used for this run"
    )

    io_config: IOConfig = Field(
        ...,
        description="I/O configuration used for this run"
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
    
    @classmethod
    def create(
        cls,
        run_name: str,
        model_config_name: str,
        prompt_config_name: str,
        retry_config_name: str,
        io_config_name: str,
        model_cfg: ModelConfig,
        prompt_config: PromptConfig,
        retry_config: RetryConfig,
        io_config: IOConfig,
        input_run_name: str,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
        **kwargs
    ) -> "InferRunInfo":
        """Create an InferRunInfo with computed deterministic UUID.

        Args:
            run_name: Run identifier (timestamp-based)
            model_config_name: Model config name
            prompt_config_name: Prompt config name
            retry_config_name: Retry config name
            io_config_name: I/O config name
            model_cfg: Full model configuration
            prompt_config: Full prompt configuration
            retry_config: Full retry configuration
            io_config: Full I/O configuration
            input_run_name: Ingest run name to read samples from
            start_idx: Start index into NormalizedDataset (None = start from beginning)
            end_idx: End index into NormalizedDataset (None = process until end)
            **kwargs: Additional fields from base RunInfo (git_sha, etc.)

        Returns:
            InferRunInfo instance with computed id
        """
        run_info_id = compute_infer_run_uuid(run_name)
        return cls(
            id=run_info_id,
            run_name=run_name,
            model_config_name=model_config_name,
            prompt_config_name=prompt_config_name,
            retry_config_name=retry_config_name,
            io_config_name=io_config_name,
            model_cfg=model_cfg,
            prompt_config=prompt_config,
            retry_config=retry_config,
            io_config=io_config,
            input_run_name=input_run_name,
            start_idx=start_idx,
            end_idx=end_idx,
            **kwargs
        )
