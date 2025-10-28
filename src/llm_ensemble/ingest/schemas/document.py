"""Document schema for IR datasets."""
from __future__ import annotations
from pydantic import BaseModel, Field


class Document(BaseModel):
    """Represents a document in an IR dataset.

    Uses external_id to clearly indicate this is the dataset's original identifier,
    not an internal system ID.
    """

    external_id: str = Field(
        ...,
        description="Document identifier from the original dataset (e.g., 'd456', 'doc_abc')"
    )
    doc_text: str = Field(..., description="The document text content")
