"""Config builder for infer CLI.

Layer 1: Configuration Assembly (Pure Data)

Responsibilities:
- Load all YAML configurations
- Create metadata-only entities (no adapter instantiation)
- Bundle into InferRunConfig (pure Pydantic data)
- Return configuration snapshot for execution

This layer knows about:
- YAML file loading
- Pydantic config models
- Metadata entity creation

This layer does NOT know about:
- Adapter classes or factories
- Concrete implementations
- Business logic
"""

from __future__ import annotations
from typing import Optional

from llm_ensemble.infer.schemas.infer_run_config import InferRunConfig
from llm_ensemble.infer.schemas.infer_run_context import IngestRunContext
from llm_ensemble.infer.schemas.model_config_schema import ModelConfig
from llm_ensemble.infer.schemas.retry_config_schema import RetryConfig
from llm_ensemble.infer.schemas.entities.provider import Provider
from llm_ensemble.infer.schemas.entities.prompt_template import PromptTemplate
from llm_ensemble.infer.schemas.entities.prompt_builder import PromptBuilder
from llm_ensemble.infer.schemas.entities.parser import Parser
from llm_ensemble.infer.adapters.template_factory import PromptTemplateFactory


def build_infer_config(
    model_config_name: str,
    provider_name: str,
    prompt_template_name: str,
    retry_config_name: str,
    input_run_name: str,
    start_idx: Optional[int] = None,
    end_idx: Optional[int] = None,
) -> InferRunConfig:
    """Build InferRunConfig from CLI arguments.

    Loads all YAML configurations and creates metadata-only entities.
    No adapter instantiation happens here - only pure data objects.

    Args:
        model_config_name: Name of model config file (e.g., "gpt-oss-20b")
        provider_name: Provider name (e.g., "openrouter", "ollama")
        prompt_template_name: Prompt template name (e.g., "thomas-simple")
        retry_config_name: Name of retry config file (e.g., "standard")
        input_run_name: Ingest run identifier for input data
        start_idx: Start index into dataset (None = from beginning)
        end_idx: End index into dataset (None = until end)

    Returns:
        InferRunConfig: Immutable configuration bundle (pure data)

    Raises:
        FileNotFoundError: If config files don't exist
        ValueError: If config is invalid
    """
    # Load configurations from YAML files
    model_config = ModelConfig.load(model_config_name)
    retry_config = RetryConfig.load(retry_config_name)

    # Create provider metadata entity (no adapter instantiation)
    provider_entity = Provider(name=provider_name)

    # Get prompt template metadata (no adapter instantiation)
    prompt_template_entity = PromptTemplateFactory.get_metadata(prompt_template_name)

    # Create execution context entity
    execution_context = IngestRunContext(
        input_run_name=input_run_name,
        start_idx=start_idx,
        end_idx=end_idx,
    )

    # Bundle into immutable config
    return InferRunConfig(
        model_cfg=model_config,
        provider=provider_entity,
        prompt_template=prompt_template_entity,
        retry_config=retry_config,
        ingest_run_context=execution_context,
    )
