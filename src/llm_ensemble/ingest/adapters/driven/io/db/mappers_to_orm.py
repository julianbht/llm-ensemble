"""Domain to ORM mappers for INGEST CLI (writing direction).

This module provides conversion functions for mapping from Pydantic domain objects
to SQLAlchemy ORM models for persistence.

Design principles:
- Domain objects are the source of truth and already have UUIDs
- Mappers are simple pass-through converters
- Stateless pure functions
- Used by DBWriter for persisting ingestion results

The domain layer works with Pydantic models (Query, Document, JudgingSample, etc.).
The persistence layer works with SQLAlchemy ORMs.
These mappers handle the impedance mismatch for the write path.
"""

from __future__ import annotations
from uuid import UUID

from llm_ensemble.ingest.domain.entities.query import Query
from llm_ensemble.ingest.domain.entities.document import Document
from llm_ensemble.ingest.domain.entities.judging_sample import JudgingSample
from llm_ensemble.ingest.domain.entities.dataset_sample import DatasetSample
from llm_ensemble.ingest.domain.entities.normalized_dataset import NormalizedDataset
from llm_ensemble.ingest.domain.entities.ingest_run_info import IngestRunInfo
from llm_ensemble.ingest.domain.entities.ingest_run_config import IngestRunConfig
from llm_ensemble.ingest.adapters.driven.io.db.orms import (
    QueryORM,
    DocumentORM,
    JudgingSampleORM,
    NormalizedDatasetORM,
    DatasetSampleORM,
    IngestRunInfoORM,
    IngestRunConfigORM,
)


# ============================================================================
# Query Mappers
# ============================================================================

def query_to_orm(query: Query) -> QueryORM:
    """Convert Query domain object to QueryORM.

    Args:
        query: Query domain object

    Returns:
        QueryORM model ready for persistence
    """
    return QueryORM(
        id=query.id,
        content_hash=query.content_hash,
        query_text=query.query_text,
    )


# ============================================================================
# Document Mappers
# ============================================================================

def document_to_orm(document: Document) -> DocumentORM:
    """Convert Document domain object to DocumentORM.

    Args:
        document: Document domain object

    Returns:
        DocumentORM model ready for persistence
    """
    return DocumentORM(
        id=document.id,
        content_hash=document.content_hash,
        doc_text=document.doc_text,
    )


# ============================================================================
# JudgingSample Mappers
# ============================================================================

def judging_sample_to_orm(sample: JudgingSample) -> JudgingSampleORM:
    """Convert JudgingSample domain object to JudgingSampleORM.

    Args:
        sample: JudgingSample domain object

    Returns:
        JudgingSampleORM model ready for persistence
    """
    return JudgingSampleORM(
        id=sample.id,
        query_id=sample.query.id,
        document_id=sample.document.id,
        gold_score=sample.gold_score,
    )


# ============================================================================
# DatasetSample Mappers
# ============================================================================

def dataset_sample_to_orm(dataset_sample: DatasetSample) -> DatasetSampleORM:
    """Convert DatasetSample domain object to DatasetSampleORM.

    Args:
        dataset_sample: DatasetSample domain object

    Returns:
        DatasetSampleORM model ready for persistence
    """
    return DatasetSampleORM(
        id=dataset_sample.id,
        normalized_dataset_id=dataset_sample.normalized_dataset_id,
        judging_sample_id=dataset_sample.judging_sample.id,
        sequence_number=dataset_sample.sequence_number,
    )


# ============================================================================
# NormalizedDataset Mappers
# ============================================================================

def normalized_dataset_to_orm(normalized_dataset: NormalizedDataset) -> NormalizedDatasetORM:
    """Convert NormalizedDataset domain object to NormalizedDatasetORM.

    Extracts the ingest_run_config_id from the embedded run_config.

    Note: This only creates the NormalizedDatasetORM entity itself.
    The junction table records (linking to samples) must be created separately.

    Args:
        normalized_dataset: NormalizedDataset domain object with embedded run_config

    Returns:
        NormalizedDatasetORM model ready for persistence
    """
    return NormalizedDatasetORM(
        id=normalized_dataset.id,
        fingerprint=normalized_dataset.fingerprint,
        external_dataset_name=normalized_dataset.external_dataset_name,
        ingest_run_config_id=normalized_dataset.run_config.id,
    )


# ============================================================================
# IngestRunConfig Mappers
# ============================================================================

def ingest_run_config_to_orm(run_config: IngestRunConfig) -> IngestRunConfigORM:
    """Convert IngestRunConfig to IngestRunConfigORM.

    Args:
        run_config: IngestRunConfig domain object

    Returns:
        IngestRunConfigORM model ready for persistence
    """
    return IngestRunConfigORM(
        id=run_config.id,
        io_config_name=run_config.io_config_name,
        input_path=run_config.input_path,
        limit=run_config.limit,
    )


# ============================================================================
# IngestRunInfo Mappers
# ============================================================================

def ingest_run_info_to_orm(run_info: IngestRunInfo) -> IngestRunInfoORM:
    """Convert IngestRunInfo aggregate to IngestRunInfoORM.

    Extracts IDs from embedded objects within the aggregate:
    - normalized_dataset.id from run_info.normalized_dataset
    - run_config.id from run_info.normalized_dataset.run_config

    Args:
        run_info: IngestRunInfo aggregate root containing embedded normalized_dataset

    Returns:
        IngestRunInfoORM model ready for persistence
    """
    return IngestRunInfoORM(
        id=run_info.id,
        run_name=run_info.run_name,
        run_type=run_info.run_type,
        normalized_dataset_id=run_info.normalized_dataset.id,
        git_sha=run_info.git_info.git_sha,
        git_branch=run_info.git_info.git_branch,
        git_is_dirty="true" if not run_info.git_info.git_clean else "false",
        notes=run_info.notes,
    )
