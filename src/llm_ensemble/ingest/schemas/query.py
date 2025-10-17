"""Query schema for IR datasets."""
from __future__ import annotations
from pydantic import BaseModel, Field


class Query(BaseModel):
    """Represents a search query in an IR dataset."""
    
    query_id: str = Field(..., description="Unique query ID")
    query_text: str = Field(..., description="The natural language query")
