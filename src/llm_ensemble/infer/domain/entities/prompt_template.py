"""PromptTemplate - metadata for a prompt template.

Pure Pydantic model bundling template text with builder/parser metadata.
"""

from __future__ import annotations
import uuid
from uuid import UUID
from pydantic import BaseModel, Field

from llm_ensemble.infer.domain.entities.prompt_builder import PromptBuilder
from llm_ensemble.infer.domain.entities.reponse_parser import ResponseParser


class PromptTemplate(BaseModel):
    """Prompt template metadata.
    
    Pure Pydantic model - bundles template text with builder/parser names."""

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

    response_parser: ResponseParser = Field(
        ...,
        description="Response parser metadata (id, name)",
        alias="response_text_parser"
    )
