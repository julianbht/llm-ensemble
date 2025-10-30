"""JudgingSample schema - a query-document pair with gold relevance score."""

from __future__ import annotations
from pydantic import BaseModel, Field

from llm_ensemble.ingest.schemas.query import Query
from llm_ensemble.ingest.schemas.document import Document
from llm_ensemble.libs.schemas import RelevanceScore  # Shared schema
from llm_ensemble.ingest.schemas.ingest_run_info import IngestRunInfo


class JudgingSample(BaseModel):
    """A single judging sample: query + document + gold relevance score + run info.

    This is the canonical normalized unit for LLM relevance judging.
    Represents a single query-document pair with its ground truth relevance label,
    along with a reference to the ingest run info (Many-to-One relationship).

    Each sample carries complete provenance metadata (via run_info) from the moment
    it's created, without waiting for aggregate statistics at the end of the run.
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

    run_info: IngestRunInfo = Field(
        ...,
        description="Reference to the ingest run info (Many-to-One relationship)"
    )
