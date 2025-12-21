"""Factory for creating InferRunConfig entities.

Domain Layer - Factory Pattern

Creates InferRunConfig from domain entities, assembling the complete
configuration bundle for manifest persistence.

This factory belongs in the domain layer because it only depends on
domain entities and performs pure assembly logic. The application layer
is responsible for extracting these entities from adapters.
"""

from __future__ import annotations

from llm_ensemble.infer.domain.entities.infer_run_config import InferRunConfig
from llm_ensemble.infer.domain.entities.ingest_run_context import IngestRunContext
from llm_ensemble.infer.domain.entities.prompt_template import PromptTemplate
from llm_ensemble.infer.domain.entities.provider import Provider
from llm_ensemble.infer.domain.entities.model_config import ModelConfig
from llm_ensemble.infer.domain.entities.retry_config_schema import RetryConfig
from llm_ensemble.infer.domain.entities.prompt_builder import PromptBuilder
from llm_ensemble.infer.domain.entities.reponse_parser import ResponseParser


class InferRunConfigFactory:
    """Factory for creating InferRunConfig entities from domain entities.

    Domain layer factory - pure assembly logic with no adapter dependencies."""

    @staticmethod
    def create(
        provider: Provider,
        model_config: ModelConfig,
        retry_config: RetryConfig,
        prompt_builder: PromptBuilder,
        response_parser: ResponseParser,
        template_text: str,
        io_name: str,
        input_run_name: str,
        start_idx: int,
        end_idx: int,
    ) -> InferRunConfig:
        """Create InferRunConfig from domain entities.

        Args:
            provider: LLM provider configuration entity
            model_config: Model configuration entity
            retry_config: Retry configuration entity
            prompt_builder: Prompt builder configuration entity
            response_parser: Response parser configuration entity
            template_text: The actual prompt template text
            io_name: I/O configuration name
            input_run_name: Resolved ingest run name
            start_idx: Actual start index used
            end_idx: Actual end index used

        Returns:
            Assembled InferRunConfig entity
        """

        # Build PromptTemplate entity
        prompt_template = PromptTemplate(
            name=prompt_builder.name,
            template_text=template_text,
            prompt_builder=prompt_builder,
            response_text_parser=response_parser,
        )

        # Build IngestRunContext entity
        ingest_run_context = IngestRunContext(
            input_run_name=input_run_name,
            start_idx=start_idx,
            end_idx=end_idx,
        )

        # Assemble complete config
        return InferRunConfig(
            model_cfg=model_config,
            retry_config=retry_config,
            prompt_template=prompt_template,
            provider=provider,
            io_name=io_name,
            ingest_run_context=ingest_run_context,
        )
