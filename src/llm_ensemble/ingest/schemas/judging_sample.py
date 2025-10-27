"""JudgingSample schema - a query-document pair with gold relevance score."""

from __future__ import annotations
from pydantic import BaseModel, Field

from llm_ensemble.ingest.schemas.query import Query
from llm_ensemble.ingest.schemas.document import Document
from llm_ensemble.ingest.schemas.relevance_score import RelevanceScore


class JudgingSample(BaseModel):
    """A single judging sample: query + document + gold relevance score.

    This is the canonical normalized unit for LLM relevance judging.
    Represents a single query-document pair with its ground truth relevance label.
    """

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
