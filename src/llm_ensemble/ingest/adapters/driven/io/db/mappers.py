"""Bidirectional mappers between domain objects and ORM models.

This module provides symmetric conversion functions for mapping between
pure Pydantic domain objects and SQLAlchemy ORM models.

Design principles:
- Bidirectional: Each entity has to_orm() and from_orm() functions
- Symmetric: Conversion logic lives in one place for both read and write
- Stateless: Pure functions with no side effects
- Explicit: Clear parameter names for foreign keys that aren't in domain objects

The domain layer works with pure Pydantic objects (Query, Document, JudgingSample).
The persistence layer works with SQLAlchemy ORMs (QueryORM, DocumentORM, etc.).
These mappers handle the impedance mismatch.

Design:
- Queries and Documents are global entities with content-based hashing
- No Dataset entity - context is tracked at NormalizedDataset level
- Content hashes are computed by domain models, not mappers
"""

from __future__ import annotations
from uuid import UUID

from llm_ensemble.ingest.domain.entities.query import Query
from llm_ensemble.ingest.domain.entities.document import Document
from llm_ensemble.ingest.domain.entities.judging_sample import JudgingSample
from llm_ensemble.ingest.domain.entities.dataset_sample import DatasetSample
from llm_ensemble.ingest.domain.entities.normalized_dataset import NormalizedDataset
from llm_ensemble.ingest.domain.entities.ingest_run_info import IngestRunInfo
from llm_ensemble.ingest.adapters.driven.io.db.orms import (
    QueryORM,
    DocumentORM,
    JudgingSampleORM,
    NormalizedDatasetORM,
    DatasetSampleORM,
    IngestRunORM,
)


# ============================================================================
# Query Mappers
# ============================================================================

def query_to_orm(query: Query) -> QueryORM:
    """Convert Query domain object to QueryORM.

    Simply maps fields - content_hash is already computed by domain model.

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


def query_from_orm(query_orm: QueryORM) -> Query:
    """Convert QueryORM to Query domain object.

    Args:
        query_orm: QueryORM model from database

    Returns:
        Query domain object
    """
    return Query(
        id=query_orm.id,
        content_hash=query_orm.content_hash,
        query_text=query_orm.query_text,
    )


# ============================================================================
# Document Mappers
# ============================================================================

def document_to_orm(document: Document) -> DocumentORM:
    """Convert Document domain object to DocumentORM.

    Simply maps fields - content_hash is already computed by domain model.

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


def document_from_orm(document_orm: DocumentORM) -> Document:
    """Convert DocumentORM to Document domain object.

    Args:
        document_orm: DocumentORM model from database

    Returns:
        Document domain object
    """
    return Document(
        id=document_orm.id,
        content_hash=document_orm.content_hash,
        doc_text=document_orm.doc_text,
    )


# ============================================================================
# JudgingSample Mappers
# ============================================================================

def judging_sample_to_orm(sample: JudgingSample) -> JudgingSampleORM:
    """Convert JudgingSample domain object to JudgingSampleORM.

    The relationship to IngestRun is now handled via the junction table,
    so ingest_run_id is no longer a field on JudgingSampleORM.

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


def judging_sample_from_orm(
    sample_orm: JudgingSampleORM,
    query: Query,
    document: Document,
) -> JudgingSample:
    """Convert JudgingSampleORM to JudgingSample domain object.

    Note: Query and Document domain objects must be provided separately as the
    JudgingSample domain object embeds them directly (not as foreign keys).

    This function expects the caller to have already loaded and converted the
    related Query and Document entities.

    Args:
        sample_orm: JudgingSampleORM model from database
        query: Query domain object (already converted from QueryORM)
        document: Document domain object (already converted from DocumentORM)

    Returns:
        JudgingSample domain object with embedded query and document
    """
    return JudgingSample(
        id=sample_orm.id,
        query=query,
        document=document,
        gold_score=sample_orm.gold_score,
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


def dataset_sample_from_orm(
    dataset_sample_orm: DatasetSampleORM,
    judging_sample: JudgingSample,
) -> DatasetSample:
    """Convert DatasetSampleORM to DatasetSample domain object.

    Args:
        dataset_sample_orm: DatasetSampleORM model from database
        judging_sample: JudgingSample domain object (already converted from ORM)

    Returns:
        DatasetSample domain object with embedded judging_sample
    """
    return DatasetSample(
        id=dataset_sample_orm.id,
        normalized_dataset_id=dataset_sample_orm.normalized_dataset_id,
        judging_sample=judging_sample,
        sequence_number=dataset_sample_orm.sequence_number,
    )


# ============================================================================
# NormalizedDataset Mappers
# ============================================================================

def normalized_dataset_to_orm(normalized_dataset: NormalizedDataset) -> NormalizedDatasetORM:
    """Convert NormalizedDataset domain object to NormalizedDatasetORM.

    Note: This only creates the NormalizedDatasetORM entity itself.
    The junction table records (linking to samples) must be created separately.

    Args:
        normalized_dataset: NormalizedDataset domain object

    Returns:
        NormalizedDatasetORM model ready for persistence
    """
    return NormalizedDatasetORM(
        id=normalized_dataset.id,
        fingerprint=normalized_dataset.fingerprint,
        external_dataset_name=normalized_dataset.external_dataset_name,
    )


def normalized_dataset_from_orm(
    normalized_dataset_orm: NormalizedDatasetORM,
    dataset_samples: list[DatasetSample],
) -> NormalizedDataset:
    """Convert NormalizedDatasetORM to NormalizedDataset domain object.

    Args:
        normalized_dataset_orm: NormalizedDatasetORM model from database
        dataset_samples: List of DatasetSample domain objects (already converted from ORM)

    Returns:
        NormalizedDataset domain object with embedded dataset samples
    """
    return NormalizedDataset(
        id=normalized_dataset_orm.id,
        fingerprint=normalized_dataset_orm.fingerprint,
        external_dataset_name=normalized_dataset_orm.external_dataset_name,
        samples=dataset_samples,
    )


# ============================================================================
# IngestRun Mappers
# ============================================================================

def ingest_run_info_to_orm(
    run_info: IngestRunInfo,
    normalized_dataset_id: UUID
) -> IngestRunORM:
    """Convert IngestRunInfo to IngestRunORM.

    Note: IngestRunInfo is a rich context object with full configuration,
    but the ORM only persists a subset of fields for run tracking.

    Args:
        run_info: IngestRunInfo context object
        normalized_dataset_id: UUID of the NormalizedDataset this run produced

    Returns:
        IngestRunORM model ready for persistence
    """
    return IngestRunORM(
        id=run_info.id,
        run_name=run_info.run_name,
        run_type=run_info.run_type,
        normalized_dataset_id=normalized_dataset_id,
        io_config_name=run_info.io_config_name,
        input_path=run_info.input_path,
        limit=run_info.limit,
        git_sha=run_info.git_info.git_sha,
        git_branch=run_info.git_info.git_branch,
        git_is_dirty="true" if not run_info.git_info.git_clean else "false",
    )
