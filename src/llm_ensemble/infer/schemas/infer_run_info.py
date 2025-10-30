"""InferRunInfo schema - extends base RunInfo with infer-specific configuration.

This contains all inference-specific configuration that is known before the run
starts and remains immutable throughout execution. By bundling this with the
base RunInfo, each LLMJudgement can carry complete provenance metadata without
waiting for the run to complete.
"""

from __future__ import annotations
from typing import Optional
from pydantic import Field

from llm_ensemble.libs.runtime.run_info import RunInfo
from llm_ensemble.libs.schemas import IOConfig
from llm_ensemble.infer.schemas.model_config_schema import ModelConfig
from llm_ensemble.infer.schemas.prompt_config_schema import PromptConfig


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

    # Override cli_name from base RunInfo to automatically set it to "infer"
    cli_name: str = Field(
        default="infer",
        description="Name of the CLI that generated this run (always 'infer' for InferRunInfo)"
    )

    # Configuration names (what user requested)
    model_config_name: str = Field(
        ...,
        description="Name of the model config used (e.g., 'gpt-oss-20b')"
    )

    prompt_config_name: str = Field(
        ...,
        description="Name of the prompt config used (e.g., 'thomas-et-al-prompt')"
    )

    io_config_name: str = Field(
        ...,
        description="Name of the I/O config used (e.g., 'ndjson')"
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

    io_config: IOConfig = Field(
        ...,
        description="I/O configuration used for this run"
    )

    # Input parameters
    input_file: str = Field(
        ...,
        description="Path to input file containing JudgingExample records"
    )

    limit: Optional[int] = Field(
        default=None,
        description="Maximum number of examples to process (None = no limit)"
    )

    # Pydantic-specific pattern to make this class immutable
    class Config:
        """Pydantic config."""
        frozen = True  # Make immutable to emphasize this is runtime context
