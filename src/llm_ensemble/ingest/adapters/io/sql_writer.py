"""SQL writer adapter for persisting judging samples to database.

Uses pure SQLAlchemy ORM models with deterministic UUIDs.
Handles its own logging and returns write summary as metadata.

This adapter delegates ORM mapping to the mappers module for bidirectional symmetry.
"""

from __future__ import annotations
from typing import List, Dict, Tuple
from uuid import UUID
import structlog

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
from llm_ensemble.libs.logging.log_events import IngestWriteEvent


class SqlWriter(DatasetWriter):
    """SQL writer adapter for judging samples - handles ORM mapping.

    Writes judging samples to SQL database using pure SQLAlchemy ORM.
    Contains the mapping layer that extracts dataset/run context from run_info
    and handles ORM relationships.

    Features:
    - Idempotent writes via merge (insert or update if exists)
    - Uses session_context() for transaction management
    - Logs write operations directly

    Database URL is read from DATABASE_URL environment variable (required).
    Example: postgresql://user:password@localhost:5432/llm_ensemble
    """

    def __init__(self, database_url: str | None = None):
        """Initialize SQL writer with its own logger."""
        self.database_url = database_url
        self.engine = get_engine(database_url)
        self.logger = structlog.get_logger().bind(component="sql_writer")

    def write(
        self,
        normalized_dataset: NormalizedDataset,
        run_info: IngestRunInfo,
    ) -> WriteSummary:
        """Write judging samples to SQL database with direct logging.

        Idempotent operation - merges entities (insert if new, update if exists).
        Logs each entity type write and summary.

        Args:
            normalized_dataset: Complete normalized dataset with samples and metadata
            run_info: Immutable runtime context

        Returns:
            WriteSummary as pure data (metadata for run summary)

        Raises:
            IOError: If database write fails
        """
        # Extract samples and dataset from normalized dataset
        samples = normalized_dataset.samples
        dataset = normalized_dataset.dataset

        if not samples:
            return WriteSummary()

        # Note: Tables must be created via `make db-init` before first write
        # Create summary builder
        summary = WriteSummary()

        # Write to database in transaction
        try:
            with session_context(self.engine) as session:
                # Save entities in dependency order, adding to summary as we go
                created, skipped = self._save_dataset(session, dataset)
                summary.add_datasets(created=created, skipped=skipped)
                if created > 0 or skipped > 0:
                    self.logger.info(IngestWriteEvent.WRITE_DATASETS, created=created, skipped=skipped)

                created, skipped = self._save_ingest_run(session, run_info)
                summary.add_runs(created=created, skipped=skipped)
                if created > 0 or skipped > 0:
                    self.logger.info(IngestWriteEvent.WRITE_RUNS, created=created, skipped=skipped)

                # Collect unique queries and documents from batch
                unique_queries, unique_documents = self._collect_unique_entities(samples)

                created, skipped = self._save_queries(session, unique_queries, dataset.id)
                summary.add_queries(created=created, skipped=skipped)
                if created > 0 or skipped > 0:
                    self.logger.info(IngestWriteEvent.WRITE_QUERIES, created=created, skipped=skipped)

                created, skipped = self._save_documents(session, unique_documents, dataset.id)
                summary.add_documents(created=created, skipped=skipped)
                if created > 0 or skipped > 0:
                    self.logger.info(IngestWriteEvent.WRITE_DOCUMENTS, created=created, skipped=skipped)

                created, skipped = self._save_samples(session, samples, run_info)
                summary.add_samples(created=created, skipped=skipped)
                if created > 0 or skipped > 0:
                    self.logger.info(IngestWriteEvent.WRITE_JUDGING_SAMPLES, created=created, skipped=skipped)

            # Log totals
            if summary.total_created > 0 or summary.total_skipped > 0:
                self.logger.info(
                    IngestWriteEvent.WRITE_COMPLETE,
                    total_created=summary.total_created,
                    total_skipped=summary.total_skipped,
                )

            return summary

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
