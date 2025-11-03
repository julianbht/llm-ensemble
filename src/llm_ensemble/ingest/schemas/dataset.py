"""Dataset schema for IR datasets."""
from __future__ import annotations
from uuid import UUID
from typing import Optional
from pydantic import BaseModel, Field

from llm_ensemble.libs.db import compute_dataset_uuid


class Dataset(BaseModel):
    """Represents an IR dataset (e.g., 'msmarco', 'trec-covid', 'llmjudge').
    
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
        
        Example:
            >>> dataset = Dataset.create("msmarco", "Microsoft Machine Reading Comprehension")
        """
        dataset_id = compute_dataset_uuid(name)
        return cls(
            id=dataset_id,
            name=name,
            description=description
        )
