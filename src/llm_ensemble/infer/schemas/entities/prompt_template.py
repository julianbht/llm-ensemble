"""PromptTemplate entity for the infer CLI.

Simple domain entity representing a prompt template.
"""

from __future__ import annotations
import uuid
from uuid import UUID
from pydantic import BaseModel, Field


class PromptTemplate(BaseModel):
    """Prompt template entity.

    Represents the template used to build prompts for LLM inference.
    """

    id: UUID = Field(
        default_factory=uuid.uuid4,
        description="Random UUID for this prompt template (upsert uses natural key)"
    )

    name: str = Field(
        ...,
        description="Prompt template name (e.g., 'thomas-simple')"
    )

    template_text: str = Field(
        ...,
        description="Raw template text (unrendered, e.g., Jinja template source)"
    )
