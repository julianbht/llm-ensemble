"""Inference pipeline runner.

Startup Layer - Composition Root (BlueZoneRunner equivalent)

Hexagonal Architecture - Composition Root Pattern:
1. Build domain configuration (InferRunConfig)
2. Build application hexagon (assemble ports and use case)
3. Build driving adapter (CLI driver)
4. Execute via driving adapter

This is the composition root where all dependency wiring happens.
NOT unit tested - tested via CLI integration tests.

Comparable to BlueZoneRunner in the hex-arch-example, which:
- Selects and instantiates adapters (driven side)
- Builds the application hexagon
- Instantiates drivers (driving side)
- Runs the drivers
"""
from __future__ import annotations

from llm_ensemble.infer.startup.adapter_config import AdapterConfig, ExecutionParams
from llm_ensemble.infer.application.inference_use_case import InferenceUseCase
from llm_ensemble.infer.domain.entities.infer_run_config import InferRunConfig
from llm_ensemble.infer.domain.entities.ingest_run_context import IngestRunContext
from llm_ensemble.infer.domain.entities.model_config import ModelConfig
from llm_ensemble.infer.domain.entities.provider import Provider
from llm_ensemble.infer.schemas.retry_config_schema import RetryConfig

from llm_ensemble.infer.adapters.io_factory import IOAdapterFactory
from llm_ensemble.infer.adapters.provider_factory import ProviderFactory
from llm_ensemble.infer.adapters.template_factory import PromptTemplateFactory
from llm_ensemble.infer.adapters.retrying_provider import RetryingProvider
from llm_ensemble.infer.adapters.cli_driver import CLIDriver

from llm_ensemble.libs.runtime.tag_manager import TagManager


def run_inference(
    adapter_config: AdapterConfig,
    execution_params: ExecutionParams,
) -> None:
    """Run inference pipeline via CLI driver (composition root).

    Hexagonal Architecture composition root pattern:
    1. Build domain configuration
    2. Build application hexagon (driven ports + use case)
    3. Build CLI driving adapter
    4. Execute via driver

    Comparable to BlueZoneRunner.main() which builds adapters,
    application, and drivers, then calls driver.run().

    Args:
        adapter_config: Specifies which adapters to instantiate
        execution_params: Execution parameters for the run
    """
    # Phase 1: Build domain configuration
    run_config = _build_run_config(adapter_config, execution_params)

    # Phase 2: Build application hexagon (driven ports + use case)
    application = _build_application(run_config)

    # Phase 3: Build CLI driving adapter
    cli_driver = CLIDriver(
        application=application,
        run_config=run_config,
        execution_params=execution_params,
        logging_config_name=adapter_config.logging_config_name,
    )

    # Phase 4: Execute via driver
    cli_driver.run()

def _build_run_config(
    adapter_config: AdapterConfig,
    execution_params: ExecutionParams,
) -> InferRunConfig:
    """Build domain configuration by loading YAML configs and instantiating config objects.

    This phase loads all configuration files and creates the domain config entity
    (InferRunConfig) which will be used to build the application.

    Args:
        adapter_config: Specifies which configs to load (by name)
        execution_params: Execution parameters for the run

    Returns:
        InferRunConfig domain entity with all loaded configurations
    """
    input_run_name = TagManager.resolve_input(execution_params.input_run_name, "ingest")

    return InferRunConfig(
        model_cfg=ModelConfig.load(adapter_config.model_config_name),
        retry_config=RetryConfig.load(adapter_config.retry_config_name),
        prompt_template=PromptTemplateFactory.create(adapter_config.prompt_template_name),
        provider=Provider(name=adapter_config.provider_name),
        io_name=adapter_config.io_name,
        ingest_run_context=IngestRunContext(
            input_run_name=input_run_name,
            start_idx=execution_params.start_idx,
            end_idx=execution_params.end_idx,
        ),
    )


def _build_application(run_config: InferRunConfig) -> InferenceUseCase:
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
        run_config: Domain configuration with loaded configs

    Returns:
        InferenceUseCase - the application's driving port interface
    """
    # Driven port: Input (reading samples)
    input_port = IOAdapterFactory.create_reader(run_config.io_name)

    # Driven port: Output (writing judgements)
    output_port = IOAdapterFactory.create_writer(run_config.io_name)

    # Driven ports: Prompt building and parsing
    template_adapter = PromptTemplateFactory.get_adapter_class(run_config.prompt_template.name)()
    prompt_builder = template_adapter.get_builder()
    response_parser = template_adapter.get_parser()

    # Driven port: LLM provider (with retry wrapper)
    base_provider = ProviderFactory.create(run_config.provider.name, run_config.model_cfg)
    llm_provider = RetryingProvider(base_provider, run_config.retry_config)

    # Assemble application hexagon (use case with driven ports)
    return InferenceUseCase(
        input_port=input_port,
        output_port=output_port,
        prompt_builder=prompt_builder,
        llm_provider=llm_provider,
        response_parser=response_parser,
    )
