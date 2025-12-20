"""Factory for creating InferRunConfig entities.

Application Layer - Factory Pattern

Creates InferRunConfig from adapter instances, extracting metadata and
assembling the complete configuration bundle for manifest persistence.

This factory belongs in the application layer because it knows about ports
and orchestrates the assembly of domain entities from adapter metadata.
"""

from __future__ import annotations

from llm_ensemble.infer.domain.entities.infer_run_config import InferRunConfig
from llm_ensemble.infer.domain.entities.ingest_run_context import IngestRunContext
from llm_ensemble.infer.domain.entities.prompt_template import PromptTemplate
from llm_ensemble.infer.application.ports.driven.llm_provider_port import LLMProviderPort
from llm_ensemble.infer.application.ports.driven.prompt_builder_port import PromptBuilderPort
from llm_ensemble.infer.application.ports.driven.response_parser_port import ResponseParserPort
from llm_ensemble.infer.application.ports.driven.output_port import OutputPort


class InferRunConfigFactory:
    """Factory for creating InferRunConfig entities from adapters.
    
    Application layer factory - knows about ports and orchestrates domain entity creation."""

    @staticmethod
    def create(
        llm_provider: LLMProviderPort,
        prompt_builder: PromptBuilderPort,
        response_parser: ResponseParserPort,
        output_port: OutputPort,
        input_run_name: str,
        start_idx: int,
        end_idx: int,
    ) -> InferRunConfig:
        """Create InferRunConfig from adapter instances."""
        # Extract metadata from adapters
        provider = llm_provider.get_provider()
        model_config = llm_provider.model_config
        retry_config = llm_provider.retry_config
        
        builder_entity = prompt_builder.get_builder()
        parser_entity = response_parser.get_parser()
        
        io_name = output_port.io_name

        # Build PromptTemplate entity (bundles template text + metadata)
        prompt_template = PromptTemplate(
            name=builder_entity.name,
            template_text=prompt_builder.get_template_text(),
            prompt_builder=builder_entity,
            response_text_parser=parser_entity,
        )

        # Assemble complete config
        return InferRunConfig(
            model_cfg=model_config,
            retry_config=retry_config,
            prompt_template=prompt_template,
            provider=provider,
            io_name=io_name,
            ingest_run_context=IngestRunContext(
                input_run_name=input_run_name,
                start_idx=start_idx,
                end_idx=end_idx,
            ),
        )
