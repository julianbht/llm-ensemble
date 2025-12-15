"""Dependency configurator for inference pipeline.

Startup Layer - Adapter Lookup & Application Building

Responsible for:
- Looking up and instantiating driven adapters (ports)
- Building the application use case with injected dependencies

Inspired by BlueZone's DependencyConfigurator pattern.
"""
from __future__ import annotations

from llm_ensemble.infer.application.inference_use_case import InferenceUseCase
from llm_ensemble.infer.domain.entities.infer_run_config import InferRunConfig
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
    instantiating adapters and building the application (hexagon).

    Uses factory pattern to lookup adapters by name/type.
    """

    def __init__(self, run_config: InferRunConfig, io_name: str):
        """Initialize dependency configurator.

        Args:
            run_config: Complete inference run configuration
            io_name: I/O adapter name (e.g., "db_to_json")
        """
        self.run_config = run_config
        self.io_name = io_name

    def lookup_input_port(self) -> InputPort:
        """Lookup and instantiate input port adapter.

        Returns:
            InputPort implementation based on io_name
        """
        return IOAdapterFactory.create_reader(self.io_name)

    def lookup_output_port(self) -> OutputPort:
        """Lookup and instantiate output port adapter.

        Returns:
            OutputPort implementation based on io_name
        """
        return IOAdapterFactory.create_writer(self.io_name)

    def lookup_prompt_builder(self) -> PromptBuilderPort:
        """Lookup and instantiate prompt builder adapter.

        Returns:
            PromptBuilderPort implementation from template
        """
        template_class = PromptTemplateFactory.get_adapter_class(
            self.run_config.prompt_template.name
        )
        template_instance = template_class()
        return template_instance.get_builder()

    def lookup_response_parser(self) -> ResponseParserPort:
        """Lookup and instantiate response parser adapter.

        Returns:
            ResponseParserPort implementation from template
        """
        template_class = PromptTemplateFactory.get_adapter_class(
            self.run_config.prompt_template.name
        )
        template_instance = template_class()
        return template_instance.get_parser()

    def lookup_llm_provider(self) -> LLMProviderPort:
        """Lookup and instantiate LLM provider adapter with retry wrapper.

        Returns:
            LLMProviderPort implementation (wrapped with retry logic)
        """
        # Create base provider
        base_provider = ProviderFactory.create(
            provider_name=self.run_config.provider.name,
            model_config=self.run_config.model_cfg,
        )

        # Wrap with retry logic
        return RetryingProvider(base_provider, self.run_config.retry_config)

    def build_application(self) -> InferenceUseCase:
        """Build application use case with all dependencies injected.

        Similar to BlueZone's buildApplication() method.

        Returns:
            InferenceUseCase with all ports injected
        """
        # Lookup all driven ports
        input_port = self.lookup_input_port()
        output_port = self.lookup_output_port()
        prompt_builder = self.lookup_prompt_builder()
        response_parser = self.lookup_response_parser()
        llm_provider = self.lookup_llm_provider()

        # Build and return application
        return InferenceUseCase(
            input_port=input_port,
            output_port=output_port,
            prompt_builder=prompt_builder,
            llm_provider=llm_provider,
            response_parser=response_parser,
        )
