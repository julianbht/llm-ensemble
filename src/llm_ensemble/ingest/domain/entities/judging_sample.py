"""JudgingSample schema - a query-document pair with gold relevance score.

This module contains the main aggregate for the ingest CLI:
- JudgingSample: The main aggregate containing a query-document pair with gold relevance

This is a pure domain entity (DTO) used for data transfer and validation.
The ORM models for persistence are separate (see orms.py).

Design:
- Queries and Documents are global entities identified by content
- No Dataset dependency - dataset context is tracked at NormalizedDataset level
"""

from __future__ import annotations
from uuid import UUID, uuid4
from pydantic import BaseModel, Field

from llm_ensemble.libs.schemas.relevance_score import RelevanceScore
from llm_ensemble.ingest.domain.entities.query import Query
from llm_ensemble.ingest.domain.entities.document import Document


class JudgingSample(BaseModel):
    """A single judging sample: query + document + gold relevance score.

    This is the canonical normalized unit for LLM relevance judging.
    Represents a single query-document pair with its ground truth relevance label.

    Contains nested Query and Document value objects as substructure.

    The id field is a random UUID (v4).

    Note: This is a pure domain entity without ORM relationships. The ORM models
    for persistence are separate (see orms.py) and handle database relationships.
    """

    id: UUID = Field(
        default_factory=uuid4,
        description="Random UUID identifier"
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
