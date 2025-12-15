"""Parser entity for the infer CLI.

Simple domain entity representing a response parser.
"""

from __future__ import annotations
import uuid
from uuid import UUID
from pydantic import BaseModel, Field


class Parser(BaseModel):
    """Response parser entity.

    Represents which parser was used to extract structured scores from LLM responses.
    """

    id: UUID = Field(
        default_factory=uuid.uuid4,
        description="Random UUID for this parser"
    )

    name: str = Field(
        ...,
        description="Parser name (e.g., 'thomas-simple')"
    )
