"""Query entity - represents a search query.

Domain entity for the ingest CLI representing a search query.
Global entity identified by content. Queries with identical text are considered
the same query regardless of which dataset they appear in.
"""

from __future__ import annotations
from uuid import UUID, uuid4
import hashlib
from pydantic import BaseModel, Field, model_validator


class Query(BaseModel):
    """Represents a search query.

    Global entity identified by content. Queries with identical text are considered
    the same query regardless of which dataset they appear in.

    The id field is a random UUID (v4).
    The content_hash is automatically computed as SHA256 hash of query text.
    """

    id: UUID = Field(
        default_factory=uuid4,
        description="Random UUID identifier"
    )
    content_hash: str = Field(
        default="",
        description="SHA256 hash of query text for content-based deduplication"
    )
    query_text: str = Field(..., description="The natural language query text")

    @model_validator(mode='after')
    def compute_content_hash(self):
        """Compute content_hash from query_text if not provided."""
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.query_text.encode()).hexdigest()
        return self
