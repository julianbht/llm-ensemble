"""SQL writer adapter for persisting judging samples to database.

Uses pure SQLAlchemy ORM models with deterministic UUIDs.
Auto-creates tables on first write and returns write summary for transparent logging.

This adapter delegates ORM mapping to the mappers module for bidirectional symmetry.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple
from uuid import UUID

from sqlalchemy.orm import Session

from llm_ensemble.ingest.schemas import JudgingSample, Dataset, Query, Document, WriteSummary, NormalizedDataset
from llm_ensemble.ingest.schemas.ingest_run_info import IngestRunInfo
from llm_ensemble.ingest.schemas.orms import (
    DatasetORM,
    QueryORM,
    DocumentORM,
    IngestRunORM,
    JudgingSampleORM,
)
from llm_ensemble.ingest.ports import DatasetWriter
from llm_ensemble.ingest.adapters.io.mappers import (
    dataset_to_orm,
    query_to_orm,
    document_to_orm,
    judging_sample_to_orm,
    ingest_run_info_to_orm,
)
from llm_ensemble.libs.db import (
    get_engine,
    session_context,
)


class SqlWriter(DatasetWriter):
    """SQL writer adapter for judging samples - handles ORM mapping.

    Writes judging samples to SQL database using pure SQLAlchemy ORM.
    Contains the mapping layer that extracts dataset/run context from run_info
    and handles ORM relationships.

    Features:
    - Auto-creates tables on first write
    - Deterministic UUIDs for all entities (computed in domain layer)
    - Central shared database across all runs for data accumulation
    - Idempotent writes via merge (insert or update if exists)
    - Uses session_context() for transaction management
    - Returns WriteSummary for transparent logging (separation of concerns)

    Mapping responsibilities:
    - Receives run_info alongside samples (contains dataset config)
    - Reconstructs Dataset entity from config for persistence
    - Maps pure domain entities (Query, Document, JudgingSample) to ORM models
    - Handles foreign key relationships at persistence time

    Database URL is read from DATABASE_URL environment variable (required).
    Example: postgresql://user:password@localhost:5432/llm_ensemble
    """

    def __init__(self, database_url: str | None = None):
        """Initialize SQL writer.

        Args:
            database_url: PostgreSQL connection URL. If None, reads from DATABASE_URL env var (required).
                         Example: postgresql://user:password@localhost:5432/llm_ensemble
        """
        self.database_url = database_url
        self.engine = get_engine(database_url)

    def write(
        self,
        normalized_dataset: NormalizedDataset,
        run_info: IngestRunInfo,
    ) -> WriteSummary:
        """Write judging samples to SQL database.

        Idempotent operation - merges entities (insert if new, update if exists).

        Args:
            normalized_dataset: Complete normalized dataset with samples and metadata
            run_info: Immutable runtime context

        Returns:
            WriteSummary tracking what was created vs. skipped

        Raises:
            IOError: If database write fails
        """
        # Extract samples and dataset from normalized dataset
        samples = normalized_dataset.samples
        dataset = normalized_dataset.dataset

        if not samples:
            return WriteSummary()

        # Note: Tables must be created via `make db-init` before first write
        # Track counts
        datasets_created = 0
        datasets_skipped = 0
        runs_created = 0
        runs_skipped = 0
        queries_created = 0
        queries_skipped = 0
        documents_created = 0
        documents_skipped = 0
        samples_created = 0
        samples_skipped = 0

        # Write to database in transaction
        try:
            with session_context(self.engine) as session:
                # Save entities in dependency order
                datasets_created, datasets_skipped = self._save_dataset(session, dataset)
                runs_created, runs_skipped = self._save_ingest_run(session, run_info)

                # Collect unique queries and documents from batch
                unique_queries, unique_documents = self._collect_unique_entities(samples)

                queries_created, queries_skipped = self._save_queries(
                    session, unique_queries, dataset.id
                )
                documents_created, documents_skipped = self._save_documents(
                    session, unique_documents, dataset.id
                )
                samples_created, samples_skipped = self._save_samples(
                    session, samples, run_info
                )

            # Return summary for orchestrator to log
            return WriteSummary(
                datasets_created=datasets_created,
                datasets_skipped=datasets_skipped,
                runs_created=runs_created,
                runs_skipped=runs_skipped,
                queries_created=queries_created,
                queries_skipped=queries_skipped,
                documents_created=documents_created,
                documents_skipped=documents_skipped,
                samples_created=samples_created,
                samples_skipped=samples_skipped,
            )

        except Exception as e:
            raise IOError(f"Failed to write samples to database: {e}") from e

    def _save_dataset(self, session: Session, dataset: Dataset) -> Tuple[int, int]:
        """Save dataset entity to database using mapper.

        Args:
            session: SQLAlchemy session
            dataset: Dataset domain object

        Returns:
            Tuple of (created_count, skipped_count)
        """
        existing = session.get(DatasetORM, dataset.id)
        if existing:
            return (0, 1)

        dataset_orm = dataset_to_orm(dataset)
        session.add(dataset_orm)
        return (1, 0)

    def _save_ingest_run(self, session: Session, run_info: IngestRunInfo) -> Tuple[int, int]:
        """Save ingest run entity to database using mapper.

        Args:
            session: SQLAlchemy session
            run_info: IngestRunInfo context object

        Returns:
            Tuple of (created_count, skipped_count)
        """
        existing = session.get(IngestRunORM, run_info.id)
        if existing:
            return (0, 1)

        ingest_run_orm = ingest_run_info_to_orm(run_info)
        session.add(ingest_run_orm)
        return (1, 0)

    def _collect_unique_entities(
        self, samples: List[JudgingSample]
    ) -> Tuple[Dict[UUID, Query], Dict[UUID, Document]]:
        """Collect unique queries and documents from samples batch.

        Args:
            samples: List of judging samples

        Returns:
            Tuple of (unique_queries_dict, unique_documents_dict) keyed by UUID
        """
        unique_queries: Dict[UUID, Query] = {}
        unique_documents: Dict[UUID, Document] = {}

        for sample in samples:
            if sample.query.id not in unique_queries:
                unique_queries[sample.query.id] = sample.query
            if sample.document.id not in unique_documents:
                unique_documents[sample.document.id] = sample.document

        return unique_queries, unique_documents

    def _save_queries(
        self, session: Session, queries: Dict[UUID, Query], dataset_id: UUID
    ) -> Tuple[int, int]:
        """Save query entities to database using mapper.

        Args:
            session: SQLAlchemy session
            queries: Dictionary of Query domain objects keyed by ID
            dataset_id: Parent dataset UUID (for foreign key)

        Returns:
            Tuple of (created_count, skipped_count)
        """
        created = 0
        skipped = 0

        for query in queries.values():
            existing = session.get(QueryORM, query.id)
            if existing:
                skipped += 1
                continue

            query_orm = query_to_orm(query, dataset_id)
            session.merge(query_orm)
            created += 1

        return (created, skipped)

    def _save_documents(
        self, session: Session, documents: Dict[UUID, Document], dataset_id: UUID
    ) -> Tuple[int, int]:
        """Save document entities to database using mapper.

        Args:
            session: SQLAlchemy session
            documents: Dictionary of Document domain objects keyed by ID
            dataset_id: Parent dataset UUID (for foreign key)

        Returns:
            Tuple of (created_count, skipped_count)
        """
        created = 0
        skipped = 0

        for document in documents.values():
            existing = session.get(DocumentORM, document.id)
            if existing:
                skipped += 1
                continue

            doc_orm = document_to_orm(document, dataset_id)
            session.merge(doc_orm)
            created += 1

        return (created, skipped)

    def _save_samples(
        self, session: Session, samples: List[JudgingSample], run_info: IngestRunInfo
    ) -> Tuple[int, int]:
        """Save judging sample entities to database using mapper.

        Args:
            session: SQLAlchemy session
            samples: List of JudgingSample domain objects
            run_info: IngestRunInfo for foreign key reference

        Returns:
            Tuple of (created_count, skipped_count)
        """
        created = 0
        skipped = 0

        for sample in samples:
            existing = session.get(JudgingSampleORM, sample.id)
            if existing:
                skipped += 1
                continue

            sample_orm = judging_sample_to_orm(sample, run_info.id)
            session.merge(sample_orm)
            created += 1

        return (created, skipped)
