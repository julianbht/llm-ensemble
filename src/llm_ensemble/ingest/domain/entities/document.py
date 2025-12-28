"""Document entity - represents a document.

Domain entity for the ingest CLI representing a document.
Global entity identified by content. Documents with identical text are considered
the same document regardless of which dataset they appear in.
"""

from __future__ import annotations
from uuid import UUID, uuid4
import hashlib
from pydantic import BaseModel, Field, model_validator


class Document(BaseModel):
    """Represents a document.

    Global entity identified by content. Documents with identical text are considered
    the same document regardless of which dataset they appear in.

    The id field is a random UUID (v4).
    The content_hash is automatically computed as SHA256 hash of document text.
    """

    id: UUID = Field(
        default_factory=uuid4,
        description="Random UUID identifier"
    )
    content_hash: str = Field(
        default="",
        description="SHA256 hash of document text for content-based deduplication"
    )
    doc_text: str = Field(..., description="The document text content")

    @model_validator(mode='after')
    def compute_content_hash(self):
        """Compute content_hash from doc_text if not provided."""
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.doc_text.encode()).hexdigest()
        return self
