"""Query schema for IR datasets."""
from __future__ import annotations
from uuid import UUID
from pydantic import BaseModel, Field

from llm_ensemble.libs.db import compute_query_uuid


class Query(BaseModel):
    """Represents a search query in an IR dataset.

    Uses external_id to clearly indicate this is the dataset's original identifier,
    not an internal system ID.
    
    The id field is a mandatory deterministic UUID computed from dataset + external_id.
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
    
    @classmethod
    def create(cls, dataset: str, external_id: str, query_text: str) -> "Query":
        """Create a Query with computed deterministic UUID.
        
        Args:
            dataset: Dataset name (e.g., 'msmarco', 'llmjudge')
            external_id: Query's external identifier
            query_text: Query text
        
        Returns:
            Query instance with computed id
        
        Example:
            >>> query = Query.create("msmarco", "q123", "what is python?")
        """
        query_id = compute_query_uuid(dataset, external_id)
        return cls(
            id=query_id,
            external_id=external_id,
            query_text=query_text
        )
