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
from llm_ensemble.ingest.schemas.normalized_dataset import NormalizedDataset
from llm_ensemble.ingest.schemas.ingest_run_info import IngestRunInfo
from llm_ensemble.ingest.schemas.orms import (
    DatasetORM,
    QueryORM,
    DocumentORM,
    JudgingSampleORM,
    NormalizedDatasetORM,
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

def query_to_orm(query: Query) -> QueryORM:
    """Convert Query domain object to QueryORM.

    Now extracts dataset_id from the embedded dataset object.

    Args:
        query: Query domain object with embedded dataset

    Returns:
        QueryORM model ready for persistence
    """
    return QueryORM(
        id=query.id,
        dataset_id=query.dataset.id,
        external_id=query.external_id,
        query_text=query.query_text,
    )


def query_from_orm(query_orm: QueryORM) -> Query:
    """Convert QueryORM to Query domain object.

    Reconstructs the embedded dataset from the ORM relationship.
    Requires the dataset relationship to be eager loaded.

    Args:
        query_orm: QueryORM model from database (with dataset relationship loaded)

    Returns:
        Query domain object with embedded dataset
    """
    dataset = dataset_from_orm(query_orm.dataset)
    return Query(
        id=query_orm.id,
        external_id=query_orm.external_id,
        query_text=query_orm.query_text,
        dataset=dataset,
    )


# ============================================================================
# Document Mappers
# ============================================================================

def document_to_orm(document: Document) -> DocumentORM:
    """Convert Document domain object to DocumentORM.

    Now extracts dataset_id from the embedded dataset object.

    Args:
        document: Document domain object with embedded dataset

    Returns:
        DocumentORM model ready for persistence
    """
    return DocumentORM(
        id=document.id,
        dataset_id=document.dataset.id,
        external_id=document.external_id,
        doc_text=document.doc_text,
    )


def document_from_orm(document_orm: DocumentORM) -> Document:
    """Convert DocumentORM to Document domain object.

    Reconstructs the embedded dataset from the ORM relationship.
    Requires the dataset relationship to be eager loaded.

    Args:
        document_orm: DocumentORM model from database (with dataset relationship loaded)

    Returns:
        Document domain object with embedded dataset
    """
    dataset = dataset_from_orm(document_orm.dataset)
    return Document(
        id=document_orm.id,
        external_id=document_orm.external_id,
        doc_text=document_orm.doc_text,
        dataset=dataset,
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
    )


def normalized_dataset_from_orm(
    normalized_dataset_orm: NormalizedDatasetORM,
    samples: list[JudgingSample],
) -> NormalizedDataset:
    """Convert NormalizedDatasetORM to NormalizedDataset domain object.

    Args:
        normalized_dataset_orm: NormalizedDatasetORM model from database
        samples: List of JudgingSample domain objects (already converted from ORM)

    Returns:
        NormalizedDataset domain object with embedded samples
    """
    return NormalizedDataset(
        id=normalized_dataset_orm.id,
        fingerprint=normalized_dataset_orm.fingerprint,
        samples=samples,
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
        git_sha=run_info.git_sha,
        git_branch=run_info.git_branch,
        git_is_dirty="true" if not run_info.git_clean else "false",
    )
