"""Ingest schemas - normalized IR dataset structures."""
from llm_ensemble.ingest.schemas.query import Query
from llm_ensemble.ingest.schemas.document import Document
from llm_ensemble.ingest.schemas.relevance import Relevance
from llm_ensemble.ingest.schemas.judging_example import JudgingExample

__all__ = [
    "Query",
    "Document", 
    "Relevance",
    "JudgingExample",
]
