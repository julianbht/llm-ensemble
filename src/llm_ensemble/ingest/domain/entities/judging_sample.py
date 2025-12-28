"""JudgingSample schema - a query-document pair with gold relevance score.

This module contains the core domain entities for the ingest CLI:
- Query: Value object representing a search query
- Document: Value object representing a document
- JudgingSample: The main aggregate containing a query-document pair with gold relevance

These are pure domain entities (DTOs) used for data transfer and validation.
The ORM models for persistence are separate (see orms.py).

Design:
- Queries and Documents are global entities identified by content
- No Dataset dependency - dataset context is tracked at NormalizedDataset level
"""

from __future__ import annotations
from uuid import UUID, uuid4
import hashlib
from pydantic import BaseModel, Field, model_validator

from llm_ensemble.libs.schemas import RelevanceScore


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


class JudgingSample(BaseModel):
    """A single judging sample: query + document + gold relevance score.

    This is the canonical normalized unit for LLM relevance judging.
    Represents a single query-document pair with its ground truth relevance label.

    Contains nested Query and Document value objects as substructure.

    The id field is a random UUID (v4).

    Note: This is a pure domain entity without ORM relationships. The ORM models
    for persistence are separate (see orms.py) and handle database relationships.
    """

    id: UUID = Field(
        default_factory=uuid4,
        description="Random UUID identifier"
    )

    query: Query = Field(
        ...,
        description="The search query"
    )

    document: Document = Field(
        ...,
        description="The document to be judged for relevance"
    )

    gold_score: RelevanceScore = Field(
        ...,
        description="Ground truth relevance score from the original dataset"
    )
