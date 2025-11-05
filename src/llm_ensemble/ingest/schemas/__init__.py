"""Ingest schemas - normalized IR dataset structures."""
from llm_ensemble.ingest.schemas.dataset import Dataset
from llm_ensemble.ingest.schemas.query import Query
from llm_ensemble.ingest.schemas.document import Document
from llm_ensemble.libs.schemas import RelevanceScore  # Shared schema
from llm_ensemble.ingest.schemas.judging_sample import JudgingSample
from llm_ensemble.ingest.schemas.ingest_run_info import IngestRunInfo
from llm_ensemble.ingest.schemas.ingest_run_summary import IngestRunSummary
from llm_ensemble.ingest.schemas.ingest_io_config import IngestIOConfig
from llm_ensemble.ingest.schemas.write_summary import WriteSummary

# ORM models (SQLAlchemy) - separate from Pydantic schemas
from llm_ensemble.ingest.schemas.orms import (
    DatasetModel,
    QueryModel,
    DocumentModel,
    IngestRunModel,
    JudgingSampleModel,
)

__all__ = [
    "Dataset",
    "Query",
    "Document",
    "RelevanceScore",
    "JudgingSample",
    "IngestRunInfo",
    "IngestRunSummary",
    "IngestIOConfig",
    "WriteSummary",
    # ORM models
    "DatasetModel",
    "QueryModel",
    "DocumentModel",
    "IngestRunModel",
    "JudgingSampleModel",
]
