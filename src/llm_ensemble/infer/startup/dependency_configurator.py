"""Dependency configuration for inference pipeline.

Startup Layer - Composition Root (Dependency Configurator)

Hexagonal Architecture - Composition Root Pattern:
Builds and wires together the application hexagon by:
1. Loading configuration from YAML files
2. Instantiating driven adapters (ports implementations)
3. Assembling the use case with its dependencies
4. Returning the application's driving port interface

This is the composition root where all dependency wiring happens.
NOT unit tested - tested via CLI integration tests.

The driving adapter (CLI) calls this configurator to build the application,
then uses the returned ForRunningInference interface to execute business logic.
"""
from __future__ import annotations

from llm_ensemble.infer.application.inference_application import InferenceApplication
from llm_ensemble.infer.application.ports.driving.for_running_inference import ForRunningInference
from llm_ensemble.infer.domain.entities.model_config import ModelConfig
from llm_ensemble.infer.schemas.retry_config_schema import RetryConfig

from llm_ensemble.infer.adapters.io_factory import IOAdapterFactory
from llm_ensemble.infer.adapters.provider_factory import ProviderFactory
from llm_ensemble.infer.adapters.template_factory import PromptTemplateFactory
from llm_ensemble.infer.adapters.retrying_provider import RetryingProvider


def build_application(
    provider_name: str,
    io_name: str,
    prompt_template_name: str,
    model_config_name: str,
    retry_config_name: str,
) -> ForRunningInference:
    """Build and wire the inference application hexagon.

    Composition root that:
    1. Loads configuration from YAML files
    2. Instantiates all driven adapters with loaded configurations
    3. Assembles the use case with its dependencies
    4. Returns the application's driving port interface

    The driving adapter (CLI) calls this function to build the application,
    then executes it via ForRunningInference.execute().

    Args:
        provider_name: Provider name (e.g., 'openrouter', 'ollama')
        io_name: I/O adapter name (e.g., 'json', 'parquet')
        prompt_template_name: Prompt template name (e.g., 'thomas-et-al')
        model_config_name: Model config name (e.g., 'gpt-oss-20b')
        retry_config_name: Retry config name (e.g., 'standard')

    Returns:
        Application implementing ForRunningInference interface
    """
    # Load configuration from YAML files
    model_cfg = ModelConfig.load(model_config_name)
    retry_cfg = RetryConfig.load(retry_config_name)

    # Build application hexagon with loaded configs
    return _build_application_hexagon(
        provider_name=provider_name,
        io_name=io_name,
        prompt_template_name=prompt_template_name,
        model_cfg=model_cfg,
        retry_cfg=retry_cfg,
    )


def _build_application_hexagon(
    provider_name: str,
    io_name: str,
    prompt_template_name: str,
    model_cfg: ModelConfig,
    retry_cfg: RetryConfig,
) -> InferenceApplication:
    """Build application hexagon by instantiating driven adapters and use case.

    This is the core of hexagonal architecture: assembling the application
    by plugging concrete adapters (driven side) into the use case.

    Driven ports (infrastructure dependencies):
    - InputPort: Reading normalized datasets
    - OutputPort: Writing LLM judgements
    - PromptBuilderPort: Building prompts from samples
    - LLMProviderPort: Calling LLM APIs
    - ResponseParserPort: Parsing LLM responses

    The use case depends only on port abstractions (ABCs), enabling:
    - Unit testing with mocked ports
    - Swapping implementations via configuration

    Args:
        provider_name: Provider name for factory
        io_name: I/O adapter name for factory
        prompt_template_name: Prompt template name for factory
        model_cfg: Loaded model configuration
        retry_cfg: Loaded retry configuration

    Returns:
        InferenceUseCase - the application's driving port interface
    """
    # Driven port: Input (reading samples)
    input_port = IOAdapterFactory.create_reader(io_name)

    # Driven port: Output (writing judgements)
    output_port = IOAdapterFactory.create_writer(io_name)

    # Driven ports: Prompt building and parsing
    template_adapter = PromptTemplateFactory.get_adapter_class(prompt_template_name)()
    prompt_builder = template_adapter.get_builder()
    response_parser = template_adapter.get_parser()

    # Driven port: LLM provider (with retry wrapper)
    base_provider = ProviderFactory.create(provider_name, model_cfg)
    llm_provider = RetryingProvider(base_provider, retry_cfg)

    # Assemble application hexagon (use case with driven ports)
    return InferenceApplication(
        input_port=input_port,
        output_port=output_port,
        prompt_builder=prompt_builder,
        llm_provider=llm_provider,
        response_parser=response_parser,
    )
