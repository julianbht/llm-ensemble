"""JudgingSample schema - a query-document pair with gold relevance score."""

from __future__ import annotations
from uuid import UUID
from pydantic import BaseModel, Field

from llm_ensemble.ingest.schemas.query import Query
from llm_ensemble.ingest.schemas.document import Document
from llm_ensemble.libs.schemas import RelevanceScore  # Shared schema
from llm_ensemble.libs.db import compute_judging_sample_uuid


class JudgingSample(BaseModel):
    """A single judging sample: query + document + gold relevance score - pure domain entity.

    This is the canonical normalized unit for LLM relevance judging.
    Represents a single query-document pair with its ground truth relevance label.

    The id field is a mandatory deterministic UUID computed from query_id + document_id.

    Note: The run_info relationship is NOT stored on the domain entity - it's only used
    during creation for UUID computation if needed and later handled at the persistence layer.
    This keeps the domain entity clean and free from ORM concerns.
    """

    id: UUID = Field(
        ...,
        description="Deterministic UUID computed from query_id + document_id"
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

    @classmethod
    def create(
        cls,
        query: Query,
        document: Document,
        gold_score: RelevanceScore,
    ) -> "JudgingSample":
        """Create a JudgingSample with computed deterministic UUID.

        Args:
            query: Query entity
            document: Document entity
            gold_score: Ground truth relevance score

        Returns:
            JudgingSample instance with computed id

        Example:
            >>> from llm_ensemble.ingest.schemas import Dataset, Query, Document, JudgingSample
            >>> from llm_ensemble.libs.schemas import RelevanceScore
            >>> dataset = Dataset.create("msmarco", "Microsoft Machine Reading Comprehension")
            >>> query = Query.create(dataset, "q123", "what is python?")
            >>> doc = Document.create(dataset, "d456", "Python is a programming language.")
            >>> sample = JudgingSample.create(query, doc, RelevanceScore.RELEVANT)
        """
        sample_id = compute_judging_sample_uuid(
            query.id,
            document.id
        )
        return cls(
            id=sample_id,
            query=query,
            document=document,
            gold_score=gold_score,
        )
