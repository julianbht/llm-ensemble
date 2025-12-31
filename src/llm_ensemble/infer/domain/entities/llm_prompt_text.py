"""LLMPromptText entity - represents a deduplicated LLM prompt.

Domain entity for the infer CLI representing prompt text sent to LLMs.
Global entity identified by content. Prompts with identical text are considered
the same prompt regardless of which judgement they appear in.
"""

from __future__ import annotations
from uuid import UUID, uuid4
import hashlib
from pydantic import BaseModel, Field, model_validator


class LLMPromptText(BaseModel):
    """Represents a deduplicated LLM prompt text.

    Global entity identified by content. Prompts with identical text are considered
    the same prompt regardless of which judgement they appear in.

    The id field is a random UUID (v4).
    The content_hash is automatically computed as SHA256 hash of prompt text.
    """

    id: UUID = Field(
        default_factory=uuid4,
        description="Random UUID identifier"
    )
    content_hash: str = Field(
        default="",
        description="SHA256 hash of prompt text for content-based deduplication"
    )
    prompt_text: str = Field(..., description="The rendered prompt text sent to the LLM")

    @model_validator(mode='after')
    def compute_content_hash(self):
        """Compute content_hash from prompt_text if not provided."""
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.prompt_text.encode()).hexdigest()
        return self
