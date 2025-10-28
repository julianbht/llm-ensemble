"""Query schema for IR datasets."""
from __future__ import annotations
from pydantic import BaseModel, Field


class Query(BaseModel):
    """Represents a search query in an IR dataset.

    Uses external_id to clearly indicate this is the dataset's original identifier,
    not an internal system ID.
    """

    external_id: str = Field(
        ...,
        description="Query identifier from the original dataset (e.g., 'q123', 'msmarco_42')"
    )
    query_text: str = Field(..., description="The natural language query text")
