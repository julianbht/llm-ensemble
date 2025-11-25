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
from uuid import UUID
import hashlib
from pydantic import BaseModel, Field

from llm_ensemble.libs.schemas import RelevanceScore
from llm_ensemble.libs.db import (
    compute_query_uuid,
    compute_document_uuid,
    compute_judging_sample_uuid,
)


class Query(BaseModel):
    """Represents a search query.

    Global entity identified by content. Queries with identical text are considered
    the same query regardless of which dataset they appear in.

    The id field is a deterministic UUID computed from query_text content.
    The content_hash is a SHA256 hash of the query text for efficient lookups.
    """

    id: UUID = Field(
        ...,
        description="Deterministic UUID computed from query_text content"
    )
    content_hash: str = Field(
        ...,
        description="SHA256 hash of query text for content-based identification"
    )
    query_text: str = Field(..., description="The natural language query text")

    @classmethod
    def create(cls, query_text: str) -> "Query":
        """Create a Query with computed deterministic UUID and content hash.

        Args:
            query_text: Query text

        Returns:
            Query instance with computed id and content_hash
        """
        content_hash = hashlib.sha256(query_text.encode()).hexdigest()
        query_id = compute_query_uuid(query_text)
        return cls(
            id=query_id,
            content_hash=content_hash,
            query_text=query_text,
        )


class Document(BaseModel):
    """Represents a document.

    Global entity identified by content. Documents with identical text are considered
    the same document regardless of which dataset they appear in.

    The id field is a deterministic UUID computed from doc_text content.
    The content_hash is a SHA256 hash of the document text for efficient lookups.
    """

    id: UUID = Field(
        ...,
        description="Deterministic UUID computed from doc_text content"
    )
    content_hash: str = Field(
        ...,
        description="SHA256 hash of document text for content-based identification"
    )
    doc_text: str = Field(..., description="The document text content")

    @classmethod
    def create(cls, doc_text: str) -> "Document":
        """Create a Document with computed deterministic UUID and content hash.

        Args:
            doc_text: Document text

        Returns:
            Document instance with computed id and content_hash
        """
        content_hash = hashlib.sha256(doc_text.encode()).hexdigest()
        doc_id = compute_document_uuid(doc_text)
        return cls(
            id=doc_id,
            content_hash=content_hash,
            doc_text=doc_text,
        )


class JudgingSample(BaseModel):
    """A single judging sample: query + document + gold relevance score.

    This is the canonical normalized unit for LLM relevance judging.
    Represents a single query-document pair with its ground truth relevance label.
    
    Contains nested Query and Document value objects as substructure.

    The id field is a mandatory deterministic UUID computed from query_id + document_id.

    Note: This is a pure domain entity without ORM relationships. The ORM models
    for persistence are separate (see orms.py) and handle database relationships.
    """

    id: UUID = Field(
        ...,
        description="Deterministic UUID computed from query_id + document_id"
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

    @classmethod
    def create(
        cls,
        query: Query,
        document: Document,
        gold_score: RelevanceScore,
    ) -> "JudgingSample":
        """Create a JudgingSample with computed deterministic UUID.

        Args:
            query: Query entity
            document: Document entity
            gold_score: Ground truth relevance score

        Returns:
            JudgingSample instance with computed id
        """
        sample_id = compute_judging_sample_uuid(
            query.id,
            document.id
        )
        return cls(
            id=sample_id,
            query=query,
            document=document,
            gold_score=gold_score,
        )
