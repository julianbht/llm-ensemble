"""Ingest schemas - normalized IR dataset structures."""
from llm_ensemble.ingest.schemas.query import Query
from llm_ensemble.ingest.schemas.document import Document
from llm_ensemble.libs.schemas import RelevanceScore  # Shared schema
from llm_ensemble.ingest.schemas.judging_sample import JudgingSample
from llm_ensemble.ingest.schemas.ingest_run_info import IngestRunInfo
from llm_ensemble.ingest.schemas.ingest_run_summary import IngestRunSummary

__all__ = [
    "Query",
    "Document",
    "RelevanceScore",
    "JudgingSample",
    "IngestRunInfo",
    "IngestRunSummary",
]
