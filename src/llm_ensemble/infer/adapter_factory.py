"""Adapter factory for infer CLI.

Layer 2: Dependency Construction (Adapter Instantiation)

Responsibilities:
- Take InferRunConfig as input
- Instantiate all concrete adapters based on config metadata
- Return InferAdapters bundle with all implementations
- This is the ONLY place that knows about adapter registries

This layer knows about:
- Adapter factory classes (ProviderFactory, IOAdapterFactory, etc.)
- Concrete adapter implementations
- Instantiation logic

This layer does NOT know about:
- CLI arguments or YAML loading (that's Layer 1)
- Execution or orchestration logic (that's Layer 3)
- Business logic (that's domain service)
"""

from __future__ import annotations
from dataclasses import dataclass

from llm_ensemble.infer.schemas.infer_run_config import InferRunConfig
from llm_ensemble.infer.ports import (
    InputPort,
    OutputPort,
    PromptBuilderPort,
    ResponseParserPort,
    LLMProviderPort,
)
from llm_ensemble.infer.adapters.provider_factory import ProviderFactory
from llm_ensemble.infer.adapters.template_factory import PromptTemplateFactory
from llm_ensemble.infer.adapters.io_factory import IOAdapterFactory
from llm_ensemble.infer.adapters.retrying_provider import RetryingProvider


@dataclass
class InferAdapters:
    """Bundle of all adapter instances needed for inference.

    This is the output of the adapter factory layer - a simple
    container of concrete implementations ready to be injected
    into the domain service.
    """

    input_adapter: InputPort
    output_adapter: OutputPort
    prompt_builder: PromptBuilderPort
    response_parser: ResponseParserPort
    llm_provider: LLMProviderPort


def build_adapters(
    config: InferRunConfig,
    io_name: str,
) -> InferAdapters:
    """Build all adapters from InferRunConfig.

    This is the single entry point for adapter instantiation.
    Takes pure config data and returns concrete implementations.

    Args:
        config: InferRunConfig with all metadata
        io_name: I/O format name (e.g., "db_to_json")

    Returns:
        InferAdapters: Bundle of all adapter instances

    Raises:
        ValueError: If any adapter lookup fails
    """
    # Instantiate I/O adapters
    input_adapter = IOAdapterFactory.create_reader(io_name)
    output_adapter = IOAdapterFactory.create_writer(io_name)

    # Instantiate prompt template adapters (builder + parser)
    template_class = PromptTemplateFactory.get_adapter_class(config.prompt_template.name)
    template_instance = template_class()
    prompt_builder = template_instance.get_builder()
    response_parser = template_instance.get_parser()

    # Instantiate base provider
    base_provider = ProviderFactory.create(
        provider_name=config.provider.name,
        model_config=config.model_cfg,
    )

    # Wrap provider with retry logic
    llm_provider = RetryingProvider(base_provider, config.retry_config)

    # Return bundled adapters
    return InferAdapters(
        input_adapter=input_adapter,
        output_adapter=output_adapter,
        prompt_builder=prompt_builder,
        response_parser=response_parser,
        llm_provider=llm_provider,
    )
