"""JudgingExample schema - the canonical normalized record for the ensemble."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel

from llm_ensemble.ingest.schemas.query import Query
from llm_ensemble.ingest.schemas.document import Document
from llm_ensemble.ingest.schemas.relevance import Relevance


class JudgingExample(BaseModel):
    """Normalized training/eval unit for the ensemble judge.

    This joins a query + document + gold label into one record so downstream
    components (prompt builders, inference adapters) can operate on a single item.
    """

    dataset: Literal["llm-judge-2024"]
    query_id: str
    query_text: str
    docid: str
    doc: str
    gold_relevance: int

    @classmethod
    def from_parts(
        cls, dataset: str, q: Query, d: Document, r: Relevance
    ) -> "JudgingExample":
        """Construct a JudgingExample from separate Query, Document, and Relevance objects."""
        assert q.query_id == r.query_id
        assert d.docid == r.docid
        return cls(
            dataset=dataset,
            query_id=q.query_id,
            query_text=q.query_text,
            docid=d.docid,
            doc=d.doc,
            gold_relevance=r.relevance,
        )
