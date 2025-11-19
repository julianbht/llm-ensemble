"""JudgingSample schema - a query-document pair with gold relevance score.

This module contains the core domain entities for the ingest CLI:
- Dataset: Value object representing an IR dataset
- Query: Value object representing a search query 
- Document: Value object representing a document
- JudgingSample: The main aggregate containing a query-document pair with gold relevance

These are pure domain entities (DTOs) used for data transfer and validation.
The ORM models for persistence are separate (see orms.py).
"""

from __future__ import annotations
from uuid import UUID
from typing import Optional
from pydantic import BaseModel, Field

from llm_ensemble.libs.schemas import RelevanceScore  # Shared schema
from llm_ensemble.libs.db import (
    compute_dataset_uuid,
    compute_query_uuid,
    compute_document_uuid,
    compute_judging_sample_uuid,
)


class Dataset(BaseModel):
    """Represents an IR dataset (e.g., 'msmarco', 'trec-covid', 'llmjudge').
    
    Value object used for UUID computation and SQL persistence.
    Each dataset represents a distinct collection of queries and documents
    for information retrieval evaluation.
    
    The id field is a mandatory deterministic UUID computed from the dataset name.
    """

    id: UUID = Field(
        ...,
        description="Deterministic UUID computed from dataset name"
    )
    name: str = Field(
        ...,
        description="Dataset name (e.g., 'msmarco', 'trec-covid', 'llmjudge')"
    )
    description: Optional[str] = Field(
        None,
        description="Optional description of the dataset"
    )
    
    @classmethod
    def create(cls, name: str, description: Optional[str] = None) -> "Dataset":
        """Create a Dataset with computed deterministic UUID.
        
        Args:
            name: Dataset name (e.g., 'msmarco', 'trec-covid')
            description: Optional dataset description
        
        Returns:
            Dataset instance with computed id
        """
        dataset_id = compute_dataset_uuid(name)
        return cls(
            id=dataset_id,
            name=name,
            description=description
        )


class Query(BaseModel):
    """Represents a search query in an IR dataset.

    Value object nested within JudgingSample. Contains the query text and external ID.
    Uses external_id to clearly indicate this is the dataset's original identifier,
    not an internal system ID.

    The id field is a mandatory deterministic UUID computed from dataset + external_id.

    The dataset is embedded as a full value object to ensure dataset context flows
    through the entire pipeline (ingest → infer → aggregate).
    """

    id: UUID = Field(
        ...,
        description="Deterministic UUID computed from dataset + external_id"
    )
    external_id: str = Field(
        ...,
        description="Query identifier from the original dataset (e.g., 'q123', 'msmarco_42')"
    )
    query_text: str = Field(..., description="The natural language query text")
    dataset: Dataset = Field(
        ...,
        description="The dataset this query belongs to"
    )

    @classmethod
    def create(cls, dataset: Dataset, external_id: str, query_text: str) -> "Query":
        """Create a Query with computed deterministic UUID.

        Args:
            dataset: Dataset entity
            external_id: Query's external identifier
            query_text: Query text

        Returns:
            Query instance with computed id
        """
        query_id = compute_query_uuid(dataset.id, external_id)
        return cls(
            id=query_id,
            external_id=external_id,
            query_text=query_text,
            dataset=dataset,
        )


class Document(BaseModel):
    """Represents a document in an IR dataset.

    Value object nested within JudgingSample. Contains the document text and external ID.
    Uses external_id to clearly indicate this is the dataset's original identifier,
    not an internal system ID.

    The id field is a mandatory deterministic UUID computed from dataset + external_id.

    The dataset is embedded as a full value object to ensure dataset context flows
    through the entire pipeline (ingest → infer → aggregate).
    """

    id: UUID = Field(
        ...,
        description="Deterministic UUID computed from dataset + external_id"
    )
    external_id: str = Field(
        ...,
        description="Document identifier from the original dataset (e.g., 'd456', 'doc_abc')"
    )
    doc_text: str = Field(..., description="The document text content")
    dataset: Dataset = Field(
        ...,
        description="The dataset this document belongs to"
    )

    @classmethod
    def create(cls, dataset: Dataset, external_id: str, doc_text: str) -> "Document":
        """Create a Document with computed deterministic UUID.

        Args:
            dataset: Dataset entity
            external_id: Document's external identifier
            doc_text: Document text

        Returns:
            Document instance with computed id
        """
        doc_id = compute_document_uuid(dataset.id, external_id)
        return cls(
            id=doc_id,
            external_id=external_id,
            doc_text=doc_text,
            dataset=dataset,
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
