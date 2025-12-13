"""PromptTemplate entity for the infer CLI.

Domain entity representing a prompt template that bundles together
a prompt builder and response parser that work as a pair.
"""

from __future__ import annotations
import uuid
from uuid import UUID
from pydantic import BaseModel, Field

from llm_ensemble.infer.schemas.entities.prompt_builder import PromptBuilder
from llm_ensemble.infer.schemas.entities.parser import Parser


class PromptTemplate(BaseModel):
    """Prompt template entity.

    Bundles together a prompt builder and response parser that are
    designed to work together. This ensures prompts and parsers are
    always correctly paired.

    The template contains metadata about both the builder (which renders
    prompts) and the parser (which extracts scores from responses).
    """

    id: UUID = Field(
        default_factory=uuid.uuid4,
        description="Random UUID for this prompt template"
    )

    name: str = Field(
        ...,
        description="Prompt template name (e.g., 'thomas-simple')"
    )

    prompt_builder: PromptBuilder = Field(
        ...,
        description="Prompt builder metadata (name, template_text)"
    )

    response_parser: Parser = Field(
        ...,
        description="Response parser metadata (name)",
        alias="response_text_parser"
    )

    @classmethod
    def create(cls, name: str, prompt_builder: PromptBuilder, response_parser: Parser) -> "PromptTemplate":
        """Create a PromptTemplate from builder and parser entities.

        Args:
            name: Template name (e.g., 'thomas-simple')
            prompt_builder: PromptBuilder metadata
            response_parser: Parser metadata

        Returns:
            PromptTemplate entity
        """
        return cls(
            name=name,
            prompt_builder=prompt_builder,
            response_text_parser=response_parser
        )

    def get_adapters(self):
        """Get adapter instances for this template.

        Returns:
            Tuple of (PromptBuilderPort, ResponseParserPort)

        Raises:
            ValueError: If template not found in factory
        """
        from llm_ensemble.infer.adapters.template_factory import PromptTemplateFactory
        return PromptTemplateFactory.create(self.name)
