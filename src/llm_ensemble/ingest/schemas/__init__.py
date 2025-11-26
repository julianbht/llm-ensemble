"""Ingest schemas - normalized IR dataset structures."""
from llm_ensemble.ingest.schemas.judging_sample import Query, Document, JudgingSample
from llm_ensemble.ingest.schemas.dataset_sample import DatasetSample
from llm_ensemble.libs.schemas import RelevanceScore  # Shared schema
from llm_ensemble.ingest.schemas.ingest_run_info import IngestRunInfo
from llm_ensemble.ingest.schemas.ingest_run_summary import IngestRunSummary
from llm_ensemble.ingest.schemas.write_summary import WriteSummary
from llm_ensemble.ingest.schemas.normalized_dataset import NormalizedDataset

# ORM models (SQLAlchemy) - separate from Pydantic schemas
from llm_ensemble.ingest.schemas.orms import (
    QueryORM,
    DocumentORM,
    NormalizedDatasetORM,
    DatasetSampleORM,
    IngestRunORM,
    JudgingSampleORM,
)

__all__ = [
    "Query",
    "Document",
    "RelevanceScore",
    "JudgingSample",
    "DatasetSample",
    "NormalizedDataset",
    "IngestRunInfo",
    "IngestRunSummary",
    "WriteSummary",
    # ORM models
    "QueryORM",
    "DocumentORM",
    "NormalizedDatasetORM",
    "DatasetSampleORM",
    "IngestRunORM",
    "JudgingSampleORM",
]
