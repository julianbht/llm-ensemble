"""Document schema for IR datasets."""
from __future__ import annotations
from pydantic import BaseModel, Field


class Document(BaseModel):
    """Represents a document in an IR dataset."""
    
    docid: str = Field(..., description="Document ID")
    doc: str = Field(..., description="Document text")
