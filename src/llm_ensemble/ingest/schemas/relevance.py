"""Relevance label schema for IR datasets."""
from __future__ import annotations
from pydantic import BaseModel, Field


class Relevance(BaseModel):
    """Represents a relevance judgment in an IR dataset."""
    
    query_id: str
    docid: str
    relevance: int = Field(..., description="Relevance label from dataset")
