"""Ingest schemas - normalized IR dataset structures."""
from llm_ensemble.ingest.schemas.judging_sample import Dataset, Query, Document, JudgingSample
from llm_ensemble.libs.schemas import RelevanceScore  # Shared schema
from llm_ensemble.ingest.schemas.ingest_run_info import IngestRunInfo
from llm_ensemble.ingest.schemas.ingest_run_summary import IngestRunSummary
from llm_ensemble.ingest.schemas.write_summary import WriteSummary
from llm_ensemble.ingest.schemas.normalized_dataset import NormalizedDataset

# ORM models (SQLAlchemy) - separate from Pydantic schemas
from llm_ensemble.ingest.schemas.orms import (
    DatasetORM,
    QueryORM,
    DocumentORM,
    IngestRunORM,
    JudgingSampleORM,
)

__all__ = [
    "Dataset",
    "Query",
    "Document",
    "RelevanceScore",
    "JudgingSample",
    "NormalizedDataset",
    "IngestRunInfo",
    "IngestRunSummary",
    "WriteSummary",
    # ORM models
    "DatasetORM",
    "QueryORM",
    "DocumentORM",
    "IngestRunORM",
    "JudgingSampleORM",
]
