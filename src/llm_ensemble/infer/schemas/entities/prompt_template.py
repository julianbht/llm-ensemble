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

    The template contains the actual template text and metadata about
    the builder and parser that process it.
    """

    id: UUID = Field(
        default_factory=uuid.uuid4,
        description="Random UUID for this prompt template"
    )

    name: str = Field(
        ...,
        description="Prompt template name (e.g., 'thomas-simple')"
    )

    template_text: str = Field(
        ...,
        description="Raw template text (unrendered)"
    )

    prompt_builder: PromptBuilder = Field(
        ...,
        description="Prompt builder metadata (id, name)"
    )

    response_parser: Parser = Field(
        ...,
        description="Response parser metadata (id, name)",
        alias="response_text_parser"
    )

    @classmethod
    def create(cls, name: str, template_text: str, prompt_builder: PromptBuilder, response_parser: Parser) -> "PromptTemplate":
        """Create a PromptTemplate from template text and metadata entities.

        Args:
            name: Template name (e.g., 'thomas-simple')
            template_text: Raw template text
            prompt_builder: PromptBuilder metadata (id, name)
            response_parser: Parser metadata (id, name)

        Returns:
            PromptTemplate entity
        """
        return cls(
            name=name,
            template_text=template_text,
            prompt_builder=prompt_builder,
            response_text_parser=response_parser
        )
