"""PromptBuilder entity for the infer CLI.

Simple metadata entity representing which prompt builder was used.
The actual template text lives on PromptTemplate.
"""

from __future__ import annotations
import uuid
from uuid import UUID
from pydantic import BaseModel, Field


class PromptBuilder(BaseModel):
    """Prompt builder entity.

    Simple metadata tracking which prompt builder was used during inference.
    Contains only identifier and name - template text lives on PromptTemplate.
    """

    id: UUID = Field(
        default_factory=uuid.uuid4,
        description="Random UUID for this prompt builder"
    )

    name: str = Field(
        ...,
        description="Prompt builder name (e.g., 'thomas-simple')"
    )
