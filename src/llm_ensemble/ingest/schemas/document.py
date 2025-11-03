"""Document schema for IR datasets."""
from __future__ import annotations
from uuid import UUID
from pydantic import BaseModel, Field

from llm_ensemble.libs.db import compute_document_uuid
from llm_ensemble.ingest.schemas.dataset import Dataset


class Document(BaseModel):
    """Represents a document in an IR dataset.

    Uses external_id to clearly indicate this is the dataset's original identifier,
    not an internal system ID.
    
    The id field is a mandatory deterministic UUID computed from dataset + external_id.
    The dataset field is a full Dataset entity reference.
    """

    id: UUID = Field(
        ...,
        description="Deterministic UUID computed from dataset + external_id"
    )
    dataset: Dataset = Field(
        ...,
        description="Dataset entity this document belongs to"
    )
    external_id: str = Field(
        ...,
        description="Document identifier from the original dataset (e.g., 'd456', 'doc_abc')"
    )
    doc_text: str = Field(..., description="The document text content")
    
    @classmethod
    def create(cls, dataset: Dataset, external_id: str, doc_text: str) -> "Document":
        """Create a Document with computed deterministic UUID.
        
        Args:
            dataset: Dataset entity
            external_id: Document's external identifier
            doc_text: Document text
        
        Returns:
            Document instance with computed id and dataset set
        
        Example:
            >>> dataset = Dataset.create("msmarco", "Microsoft Machine Reading Comprehension")
            >>> doc = Document.create(dataset, "d456", "Python is a programming language.")
        """
        doc_id = compute_document_uuid(dataset.name, external_id)
        return cls(
            id=doc_id,
            dataset=dataset,
            external_id=external_id,
            doc_text=doc_text
        )
