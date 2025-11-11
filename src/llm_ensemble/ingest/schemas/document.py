"""Document schema for IR datasets."""
from __future__ import annotations
from uuid import UUID
from pydantic import BaseModel, Field

from llm_ensemble.libs.db import compute_document_uuid
from llm_ensemble.ingest.schemas.dataset import Dataset


class Document(BaseModel):
    """Represents a document in an IR dataset - pure domain entity.

    Uses external_id to clearly indicate this is the dataset's original identifier,
    not an internal system ID.

    The id field is a mandatory deterministic UUID computed from dataset + external_id.

    Note: The dataset relationship is NOT stored on the domain entity - it's only used
    during creation for UUID computation and later handled at the persistence layer.
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

    @classmethod
    def create(cls, dataset: Dataset, external_id: str, doc_text: str) -> "Document":
        """Create a Document with computed deterministic UUID.

        Args:
            dataset: Dataset entity (used for UUID computation only, not stored)
            external_id: Document's external identifier
            doc_text: Document text

        Returns:
            Document instance with computed id

        Example:
            >>> dataset = Dataset.create("msmarco", "Microsoft Machine Reading Comprehension")
            >>> doc = Document.create(dataset, "d456", "Python is a programming language.")
        """
        doc_id = compute_document_uuid(dataset.id, external_id)
        return cls(
            id=doc_id,
            external_id=external_id,
            doc_text=doc_text,
        )
