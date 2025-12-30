"""InferRunConfig - immutable configuration bundle for an infer run.

Pure Pydantic model containing all configuration needed to execute inference:
- Model configuration (which model to use)
- Provider configuration (which LLM service to use)
- Prompt template configuration (prompt builder and parser pair)
- Retry configuration
- Execution context (input source, sample range) - INLINED
- I/O adapter name

This is a serializable domain entity (frozen Pydantic model).
Separate from InferRunInfo (git info, timestamps, run metadata).
"""

from __future__ import annotations
from typing import Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, ConfigDict, Field

from llm_ensemble.infer.domain.entities.model_config import ModelConfig
from llm_ensemble.infer.domain.entities.prompt_template import PromptTemplate
from llm_ensemble.infer.domain.entities.provider import Provider
from llm_ensemble.infer.domain.entities.retry_config_schema import RetryConfig


class InferRunConfig(BaseModel):
    """Immutable configuration bundle for an infer run.

    Pure Pydantic model - no business logic, just data."""

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

    # Execution context (inlined from IngestRunContext)
    input_run_name: str = Field(
        ...,
        description="Ingest run name to read samples from (e.g., 'my_ingest_run')"
    )

    start_idx: Optional[int] = Field(
        default=None,
        description="Start index into NormalizedDataset.samples (None = from beginning)"
    )

    end_idx: Optional[int] = Field(
        default=None,
        description="End index into NormalizedDataset.samples (None = until end)"
    )

    io_name: str = Field(
        ...,
        description="I/O adapter name (e.g., 'json', 'parquet')"
    )

    model_config = ConfigDict(frozen=True)
