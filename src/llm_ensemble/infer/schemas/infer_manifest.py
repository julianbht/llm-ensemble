"""InferManifest schema - extends base Manifest with infer-specific execution parameters."""

from __future__ import annotations
from typing import Optional
from pydantic import Field

from llm_ensemble.libs.runtime.manifest import Manifest
from llm_ensemble.infer.schemas.model_config_schema import ModelConfig
from llm_ensemble.infer.schemas.prompt_config_schema import PromptConfig
from llm_ensemble.infer.schemas.io_config_schema import IOConfig


class InferManifest(Manifest):
    """Manifest for infer CLI runs.

    Extends the base Manifest with infer-specific execution parameters:
    what the user requested and what configs were used.
    """

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

    # Output statistics (set by service at end of run)
    judgement_count: Optional[int] = Field(
        default=None,
        description="Number of judgements produced (set at end of run)"
    )

    error_count: Optional[int] = Field(
        default=None,
        description="Number of failed judgements (label=None)"
    )

    total_latency_ms: Optional[float] = Field(
        default=None,
        description="Total latency across all judgements in milliseconds"
    )

    avg_latency_ms: Optional[float] = Field(
        default=None,
        description="Average latency per judgement in milliseconds"
    )
