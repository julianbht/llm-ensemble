"""LLMResponseText entity - represents a deduplicated LLM response.

Domain entity for the infer CLI representing raw response text from LLMs.
Global entity identified by content. Responses with identical text are considered
the same response regardless of which judgement they appear in.
"""

from __future__ import annotations
from uuid import UUID, uuid4
import hashlib
from pydantic import BaseModel, Field, model_validator


class LLMResponseText(BaseModel):
    """Represents a deduplicated LLM response text.

    Global entity identified by content. Responses with identical text are considered
    the same response regardless of which judgement they appear in.

    The id field is a random UUID (v4).
    The content_hash is automatically computed as SHA256 hash of response text.
    """

    id: UUID = Field(
        default_factory=uuid4,
        description="Random UUID identifier"
    )
    content_hash: str = Field(
        default="",
        description="SHA256 hash of response text for content-based deduplication"
    )
    llm_response_text: str = Field(..., description="The raw LLM response text")

    @model_validator(mode='after')
    def compute_content_hash(self):
        """Compute content_hash from llm_response_text if not provided."""
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.llm_response_text.encode()).hexdigest()
        return self
