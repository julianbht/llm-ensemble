"""InferRunConfig - configuration bundle for an infer run.

This entity bundles all configuration needed to execute inference:
- Model configuration (which model to use)
- Provider configuration (which LLM service to use)
- Prompt template configuration (prompt builder and parser pair)
- Retry configuration
- Execution context (input source, sample range, I/O format)
- Name hints (for run_name generation)

Responsibilities:
- Provides name hints for run_name generation
- Bundles all configs needed for inference execution
- Includes execution context as a nested entity
- Immutable configuration snapshot used to produce judgements
- Can instantiate adapters via factory methods

This is separate from:
- InferRunInfo: Git info, timestamps, run metadata
- InferRunOutput: The actual judgements and metrics produced
"""

from __future__ import annotations
from uuid import UUID, uuid4
from pydantic import BaseModel, ConfigDict, Field

from llm_ensemble.infer.domain.entities.model_config import ModelConfig
from llm_ensemble.infer.domain.entities.prompt_template import PromptTemplate
from llm_ensemble.infer.domain.entities.provider import Provider
from llm_ensemble.infer.schemas.retry_config_schema import RetryConfig
from llm_ensemble.infer.domain.entities.ingest_run_context import IngestRunContext


class InferRunConfig(BaseModel):
    """Configuration bundle for an infer run.

    Contains all configuration needed to execute inference:
    - Model configuration (model ID, parameters)
    - Provider configuration (LLM service name)
    - Prompt template (bundles builder + parser metadata)
    - Retry configuration (backoff, max attempts)
    - Execution context (input source, sample range, I/O format)

    This represents "what configuration was used" and "how it was executed"
    to produce judgements. Separate from run metadata (InferRunInfo).

    Adapters can be instantiated from this config using factory methods.
    """

    id: UUID = Field(
        default_factory=uuid4,
        description="Random UUID for this config bundle"
    )

    model_cfg: ModelConfig = Field(
        ...,
        description="Model configuration (model ID, parameters)"
    )

    provider: Provider = Field(
        ...,
        description="Provider metadata (name)"
    )

    prompt_template: PromptTemplate = Field(
        ...,
        description="Prompt template (bundles builder + parser metadata)"
    )

    retry_config: RetryConfig = Field(
        ...,
        description="Retry configuration (backoff, max attempts)"
    )

    ingest_run_context: IngestRunContext = Field(
        ...,
        description="Execution context of ingest run"
    )

    model_config = ConfigDict(frozen=True)

    def get_name_hints(self) -> list[str]:
        """Get name hints for run_name generation.

        Returns:
            List of name components for run_name generation:
            [model_hint, template_name, provider_name]

        Example:
            ["gpt-oss-20b", "thomas-simple", "openrouter"]
        """
        return [
            self.model_cfg.name_hint,
            self.prompt_template.name,
            self.provider.name,
        ]
