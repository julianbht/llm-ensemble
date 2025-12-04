"""Provider entity for the infer CLI.

Simple domain entity representing an LLM provider.
"""

from __future__ import annotations
import uuid
from uuid import UUID
from pydantic import BaseModel, Field


class Provider(BaseModel):
    """LLM provider entity.

    Represents which service/platform was used to run inference
    (e.g., openrouter, ollama, hf).
    """

    id: UUID = Field(
        default_factory=uuid.uuid4,
        description="Random UUID for this provider (upsert uses natural key)"
    )

    name: str = Field(
        ...,
        description="Provider name (e.g., 'openrouter', 'ollama', 'hf')"
    )
