"""InferRunConfig - configuration bundle for an infer run.

This entity bundles all configuration needed to execute inference:
- Model configuration (which model to use)
- Adapter configuration (prompt builder, parser, provider)
- Retry configuration
- Name hints (for run_name generation)

Responsibilities:
- Provides name hints for run_name generation
- Bundles all configs needed for inference execution
- Immutable configuration snapshot used to produce judgements

This is separate from:
- InferRunInfo: Git info, timestamps, run metadata
- InferRunContext: CLI args like start_idx, end_idx, input_run_name
- InferRunOutput: The actual judgements and metrics produced
"""

from __future__ import annotations
from uuid import UUID, uuid4
from pydantic import BaseModel, ConfigDict, Field

from llm_ensemble.infer.schemas.model_config_schema import ModelConfig
from llm_ensemble.infer.schemas.entities.adapter_config import AdapterConfig
from llm_ensemble.infer.schemas.retry_config_schema import RetryConfig


class InferRunConfig(BaseModel):
    """Configuration bundle for an infer run.

    Contains all configuration needed to execute inference:
    - Model configuration (model ID, parameters)
    - Adapter configuration (prompt builder, parser, provider)
    - Retry configuration (backoff, max attempts)

    This represents "what configuration was used" to produce judgements.
    Separate from run metadata (InferRunInfo) and execution context (InferRunContext).
    """

    id: UUID = Field(
        default_factory=uuid4,
        description="Random UUID for this config bundle"
    )

    model_cfg: ModelConfig = Field(
        ...,
        description="Model configuration (model ID, parameters)"
    )

    adapter_config: AdapterConfig = Field(
        ...,
        description="Adapter configuration (prompt builder, parser, provider)"
    )

    retry_config: RetryConfig = Field(
        ...,
        description="Retry configuration (backoff, max attempts)"
    )

    model_config = ConfigDict(frozen=True)

    def get_name_hints(self) -> list[str]:
        """Get name hints for run_name generation.

        Returns:
            List of name components for run_name generation:
            [model_hint, prompt_hint, parser_hint, provider_hint]

        Example:
            ["gpt-oss-20b", "thomas-simple", "thomas-simple", "openrouter"]
        """
        return [
            self.model_cfg.hint,
            self.adapter_config.prompt_builder.name,
            self.adapter_config.parser.name,
            self.adapter_config.provider.name,
        ]
