"""Bidirectional mappers between domain objects and ORM models.

This module provides symmetric conversion functions for mapping between
pure Pydantic domain objects and SQLAlchemy ORM models.

Design principles:
- Bidirectional: Each entity has to_orm() and from_orm() functions
- Symmetric: Conversion logic lives in one place for both read and write
- Stateless: Pure functions with no side effects
- Explicit: Clear parameter names for foreign keys that aren't in domain objects

The domain layer works with pure Pydantic objects (Dataset, Query, Document, JudgingSample).
The persistence layer works with SQLAlchemy ORMs (DatasetORM, QueryORM, etc.).
These mappers handle the impedance mismatch.
"""

from __future__ import annotations
from uuid import UUID

from llm_ensemble.ingest.schemas import Dataset, Query, Document, JudgingSample
from llm_ensemble.ingest.schemas.ingest_run_info import IngestRunInfo
from llm_ensemble.ingest.schemas.orms import (
    DatasetORM,
    QueryORM,
    DocumentORM,
    JudgingSampleORM,
    IngestRunORM,
)


# ============================================================================
# Dataset Mappers
# ============================================================================

def dataset_to_orm(dataset: Dataset) -> DatasetORM:
    """Convert Dataset domain object to DatasetORM.

    Args:
        dataset: Dataset domain object

    Returns:
        DatasetORM model ready for persistence
    """
    return DatasetORM(
        id=dataset.id,
        name=dataset.name,
        description=dataset.description,
    )


def dataset_from_orm(dataset_orm: DatasetORM) -> Dataset:
    """Convert DatasetORM to Dataset domain object.

    Args:
        dataset_orm: DatasetORM model from database

    Returns:
        Dataset domain object
    """
    return Dataset(
        id=dataset_orm.id,
        name=dataset_orm.name,
        description=dataset_orm.description,
    )


# ============================================================================
# Query Mappers
# ============================================================================

def query_to_orm(query: Query, dataset_id: UUID) -> QueryORM:
    """Convert Query domain object to QueryORM.

    Note: dataset_id must be provided explicitly as it's not stored in the domain object.
    The Query domain object only knows its own ID (which is computed from dataset_id + external_id),
    but the ORM needs the foreign key.

    Args:
        query: Query domain object
        dataset_id: Parent dataset UUID (for foreign key)

    Returns:
        QueryORM model ready for persistence
    """
    return QueryORM(
        id=query.id,
        dataset_id=dataset_id,
        external_id=query.external_id,
        query_text=query.query_text,
    )


def query_from_orm(query_orm: QueryORM) -> Query:
    """Convert QueryORM to Query domain object.

    Note: The dataset relationship is not reconstructed as Query domain object
    doesn't store it. The dataset_id in the ORM is used only for foreign key constraints.

    Args:
        query_orm: QueryORM model from database

    Returns:
        Query domain object (without dataset reference)
    """
    return Query(
        id=query_orm.id,
        external_id=query_orm.external_id,
        query_text=query_orm.query_text,
    )


# ============================================================================
# Document Mappers
# ============================================================================

def document_to_orm(document: Document, dataset_id: UUID) -> DocumentORM:
    """Convert Document domain object to DocumentORM.

    Note: dataset_id must be provided explicitly as it's not stored in the domain object.
    The Document domain object only knows its own ID (which is computed from dataset_id + external_id),
    but the ORM needs the foreign key.

    Args:
        document: Document domain object
        dataset_id: Parent dataset UUID (for foreign key)

    Returns:
        DocumentORM model ready for persistence
    """
    return DocumentORM(
        id=document.id,
        dataset_id=dataset_id,
        external_id=document.external_id,
        doc_text=document.doc_text,
    )


def document_from_orm(document_orm: DocumentORM) -> Document:
    """Convert DocumentORM to Document domain object.

    Note: The dataset relationship is not reconstructed as Document domain object
    doesn't store it. The dataset_id in the ORM is used only for foreign key constraints.

    Args:
        document_orm: DocumentORM model from database

    Returns:
        Document domain object (without dataset reference)
    """
    return Document(
        id=document_orm.id,
        external_id=document_orm.external_id,
        doc_text=document_orm.doc_text,
    )


# ============================================================================
# JudgingSample Mappers
# ============================================================================

def judging_sample_to_orm(sample: JudgingSample, ingest_run_id: UUID) -> JudgingSampleORM:
    """Convert JudgingSample domain object to JudgingSampleORM.

    Note: ingest_run_id must be provided explicitly as it's not stored in the domain object.
    The JudgingSample domain object is a pure query+document+score entity, but the ORM
    tracks which ingest run created it.

    Args:
        sample: JudgingSample domain object
        ingest_run_id: Ingest run UUID (for foreign key)

    Returns:
        JudgingSampleORM model ready for persistence
    """
    return JudgingSampleORM(
        id=sample.id,
        query_id=sample.query.id,
        document_id=sample.document.id,
        ingest_run_id=ingest_run_id,
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
# IngestRun Mappers
# ============================================================================

def ingest_run_info_to_orm(run_info: IngestRunInfo) -> IngestRunORM:
    """Convert IngestRunInfo to IngestRunORM.

    Note: IngestRunInfo is a rich context object with full configuration,
    but the ORM only persists a subset of fields for run tracking.

    Args:
        run_info: IngestRunInfo context object

    Returns:
        IngestRunORM model ready for persistence
    """
    return IngestRunORM(
        id=run_info.id,
        run_name=run_info.run_name,
        run_type=run_info.run_type,
        io_config_name=run_info.io_config_name,
        input_path=run_info.input_path,
        limit=run_info.limit,
        git_sha=run_info.git_sha,
        git_branch=run_info.git_branch,
        git_is_dirty="true" if not run_info.git_clean else "false",
    )
