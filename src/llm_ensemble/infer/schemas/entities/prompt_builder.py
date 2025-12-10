"""PromptBuilder entity for the infer CLI.

Represents a prompt builder configuration (name + template text).
Renamed from PromptTemplate to match ORM naming (PromptBuilderORM).
"""

from __future__ import annotations
import uuid
from uuid import UUID
from pydantic import BaseModel, Field


class PromptBuilder(BaseModel):
    """Prompt builder entity.

    Represents the prompt builder configuration including template metadata.
    Used to track which builder created prompts during inference.
    """

    id: UUID = Field(
        default_factory=uuid.uuid4,
        description="Random UUID for this prompt builder"
    )

    name: str = Field(
        ...,
        description="Prompt builder name (e.g., 'thomas-simple')"
    )

    template_text: str = Field(
        ...,
        description="Raw template text (unrendered, e.g., Jinja template source)"
    )
