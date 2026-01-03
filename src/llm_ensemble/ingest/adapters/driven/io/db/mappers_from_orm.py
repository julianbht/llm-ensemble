"""ORM to Domain mappers for INGEST CLI (reading direction).

This module provides conversion functions for mapping from SQLAlchemy ORM models
to Pydantic domain objects.

Design principles:
- ORM models are the source when reading from database
- Mappers are simple pass-through converters
- Stateless pure functions
- Used by DBReader for loading historical runs

The domain layer works with Pydantic models (Query, Document, JudgingSample, etc.).
The persistence layer works with SQLAlchemy ORMs.
These mappers handle the impedance mismatch for the read path.
"""

from __future__ import annotations

from llm_ensemble.ingest.domain.entities.query import Query
from llm_ensemble.ingest.domain.entities.document import Document
from llm_ensemble.ingest.domain.entities.judging_sample import JudgingSample
from llm_ensemble.ingest.domain.entities.dataset_sample import NormalizedDatasetJudgingSample
from llm_ensemble.ingest.domain.entities.normalized_dataset import NormalizedDataset
from llm_ensemble.ingest.domain.entities.ingest_run import IngestRun
from llm_ensemble.ingest.domain.entities.ingest_run_config import IngestRunConfig
from llm_ensemble.ingest.adapters.driven.io.db.orms import (
    QueryORM,
    DocumentORM,
    JudgingSampleORM,
    NormalizedDatasetORM,
    NormalizedDatasetJudgingSampleORM,
    IngestRunORM,
    IngestRunConfigORM,
)


# ============================================================================
# Query Mappers
# ============================================================================

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

def dataset_sample_from_orm(
    dataset_sample_orm: NormalizedDatasetJudgingSampleORM,
    judging_sample: JudgingSample,
) -> NormalizedDatasetJudgingSample:
    """Convert NormalizedDatasetJudgingSampleORM to NormalizedDatasetJudgingSample domain object.

    Args:
        dataset_sample_orm: NormalizedDatasetJudgingSampleORM model from database
        judging_sample: JudgingSample domain object (already converted from ORM)

    Returns:
        NormalizedDatasetJudgingSample domain object with embedded judging_sample
    """
    return NormalizedDatasetJudgingSample(
        id=dataset_sample_orm.id,
        normalized_dataset_id=dataset_sample_orm.normalized_dataset_id,
        judging_sample=judging_sample,
        sequence_number=dataset_sample_orm.sequence_number,
    )


# ============================================================================
# NormalizedDataset Mappers
# ============================================================================

def normalized_dataset_from_orm(
    normalized_dataset_orm: NormalizedDatasetORM,
    dataset_samples: list[NormalizedDatasetJudgingSample],
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
# IngestRunConfig Mappers
# ============================================================================

def ingest_run_config_from_orm(run_config_orm: IngestRunConfigORM) -> IngestRunConfig:
    """Convert IngestRunConfigORM to IngestRunConfig domain object.

    Args:
        run_config_orm: IngestRunConfigORM model from database

    Returns:
        IngestRunConfig domain object
    """
    return IngestRunConfig(
        id=run_config_orm.id,
        io_config_name=run_config_orm.io_config_name,
        input_path=run_config_orm.input_path,
        limit=run_config_orm.limit,
    )


# ============================================================================
# IngestRun Mappers
# ============================================================================

def ingest_run_from_orm(
    ingest_run_orm: IngestRunORM,
    ingest_run_config: IngestRunConfig,
    normalized_dataset: NormalizedDataset,
) -> IngestRun:
    """Convert IngestRunORM to IngestRun domain object.

    Args:
        ingest_run_orm: IngestRunORM model from database
        ingest_run_config: IngestRunConfig domain object (already converted from ORM)
        normalized_dataset: NormalizedDataset domain object (already converted from ORM)

    Returns:
        IngestRun domain object
    """
    from llm_ensemble.libs.runtime.git_utils import GitInfo

    return IngestRun(
        id=ingest_run_orm.id,
        run_name=ingest_run_orm.run_name,
        run_type=ingest_run_orm.run_type,
        ingest_run_config=ingest_run_config,
        normalized_dataset=normalized_dataset,
        start_time=ingest_run_orm.start_time,
        end_time=ingest_run_orm.end_time,
        git_info=GitInfo(
            git_sha=ingest_run_orm.git_sha,
            git_branch=ingest_run_orm.git_branch,
            git_clean=ingest_run_orm.git_is_dirty != "true",
        ),
        notes=ingest_run_orm.notes,
    )
