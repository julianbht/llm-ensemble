"""Dependency configurator for inference pipeline.

Startup Layer - Adapter Lookup & Application Building

Responsible for:
- Looking up and instantiating driven adapters (ports) using factories
- Building the application use case with explicitly injected dependencies

Inspired by BlueZone's DependencyConfigurator pattern.

Key principle: Takes lightweight configs (pure data), uses factories to instantiate
adapters, and requires EXPLICIT port arguments for build_application().
"""
from __future__ import annotations

from llm_ensemble.infer.application.inference_use_case import InferenceUseCase
from llm_ensemble.infer.domain.entities.model_config import ModelConfig
from llm_ensemble.infer.domain.entities.prompt_template import PromptTemplate
from llm_ensemble.infer.schemas.retry_config_schema import RetryConfig
from llm_ensemble.infer.ports.input_port import InputPort
from llm_ensemble.infer.ports.output_port import OutputPort
from llm_ensemble.infer.ports.prompt_builder_port import PromptBuilderPort
from llm_ensemble.infer.ports.response_parser_port import ResponseParserPort
from llm_ensemble.infer.ports.llm_provider_port import LLMProviderPort

from llm_ensemble.infer.adapters.io_factory import IOAdapterFactory
from llm_ensemble.infer.adapters.provider_factory import ProviderFactory
from llm_ensemble.infer.adapters.template_factory import PromptTemplateFactory
from llm_ensemble.infer.adapters.retrying_provider import RetryingProvider


class DependencyConfigurator:
    """Adapter lookup and application building.

    Similar to BlueZone's DependencyConfigurator - responsible for
    instantiating adapters using factories and building the application (hexagon).

    Takes lightweight configs (just data, no methods) and uses them to lookup
    adapters from factories.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        provider_name: str,
        prompt_template: PromptTemplate,
        retry_config: RetryConfig,
    ):
        """Initialize dependency configurator with lightweight configs.

        Args:
            model_config: Model configuration (model ID, parameters)
            provider_name: Provider name (e.g., "openrouter", "ollama")
            prompt_template: Prompt template metadata
            retry_config: Retry configuration (backoff, max attempts)
        """
        self.model_config = model_config
        self.provider_name = provider_name
        self.prompt_template = prompt_template
        self.retry_config = retry_config

    def lookup_input_port(self, io_name: str) -> InputPort:
        """Lookup and instantiate input port adapter.

        Args:
            io_name: I/O adapter name (e.g., "db_to_json")

        Returns:
            InputPort implementation
        """
        return IOAdapterFactory.create_reader(io_name)

    def lookup_output_port(self, io_name: str) -> OutputPort:
        """Lookup and instantiate output port adapter.

        Args:
            io_name: I/O adapter name (e.g., "db_to_json")

        Returns:
            OutputPort implementation
        """
        return IOAdapterFactory.create_writer(io_name)

    def lookup_prompt_builder(self) -> PromptBuilderPort:
        """Lookup and instantiate prompt builder adapter.

        Uses prompt_template config to find the right template adapter.

        Returns:
            PromptBuilderPort implementation
        """
        template_class = PromptTemplateFactory.get_adapter_class(self.prompt_template.name)
        template_instance = template_class()
        return template_instance.get_builder()

    def lookup_response_parser(self) -> ResponseParserPort:
        """Lookup and instantiate response parser adapter.

        Uses prompt_template config to find the right template adapter.

        Returns:
            ResponseParserPort implementation
        """
        template_class = PromptTemplateFactory.get_adapter_class(self.prompt_template.name)
        template_instance = template_class()
        return template_instance.get_parser()

    def lookup_llm_provider(self) -> LLMProviderPort:
        """Lookup and instantiate LLM provider adapter with retry wrapper.

        Uses provider_name and model_config to create provider, then wraps with retry logic.

        Returns:
            LLMProviderPort implementation (wrapped with retry logic)
        """
        # Create base provider
        base_provider = ProviderFactory.create(
            provider_name=self.provider_name,
            model_config=self.model_config,
        )

        # Wrap with retry logic
        return RetryingProvider(base_provider, self.retry_config)

    def build_application(
        self,
        input_port: InputPort,
        output_port: OutputPort,
        prompt_builder: PromptBuilderPort,
        response_parser: ResponseParserPort,
        llm_provider: LLMProviderPort,
    ) -> InferenceUseCase:
        """Build application use case with explicitly injected dependencies.

        Similar to BlueZone's buildApplication() method.

        EXPLICIT arguments make dependencies VISIBLE at call site.

        Args:
            input_port: Input adapter instance
            output_port: Output adapter instance
            prompt_builder: Prompt builder adapter instance
            response_parser: Response parser adapter instance
            llm_provider: LLM provider adapter instance (with retry wrapper)

        Returns:
            InferenceUseCase with all ports injected
        """
        return InferenceUseCase(
            input_port=input_port,
            output_port=output_port,
            prompt_builder=prompt_builder,
            llm_provider=llm_provider,
            response_parser=response_parser,
        )
