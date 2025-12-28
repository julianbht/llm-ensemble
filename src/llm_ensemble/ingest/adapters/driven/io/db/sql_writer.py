"""SQL writer adapter for persisting judging samples to database.

Uses pure SQLAlchemy ORM models with random UUIDs.
Duplicate detection via database constraint violations (IntegrityError).
Handles its own logging and returns write summary as metadata.

This adapter delegates ORM mapping to the mappers module for bidirectional symmetry.
"""

from __future__ import annotations
from typing import List, Dict, Tuple
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from llm_ensemble.ingest.domain.entities import (
    JudgingSample,
    Query,
    Document,
    WriteSummary,
    NormalizedDataset,
)
from llm_ensemble.ingest.domain.entities.dataset_sample import DatasetSample
from llm_ensemble.ingest.domain.entities.ingest_run_info import IngestRunInfo
from llm_ensemble.ingest.adapters.driven.io.db.orms import (
    QueryORM,
    DocumentORM,
    IngestRunORM,
    JudgingSampleORM,
    NormalizedDatasetORM,
    DatasetSampleORM,
)
from llm_ensemble.ingest.application.ports.driven.for_output import ForOutput
from llm_ensemble.ingest.adapters.driven.io.db.mappers import (
    query_to_orm,
    document_to_orm,
    judging_sample_to_orm,
    normalized_dataset_to_orm,
    ingest_run_info_to_orm,
)
from llm_ensemble.libs.logging import get_logger
from llm_ensemble.libs.db import (
    get_engine,
    session_context,
)
from llm_ensemble.libs.logging.log_events import IngestWriteEvent


class SqlWriter(ForOutput):
    """SQL writer adapter for judging samples - handles ORM mapping.

    Writes judging samples to SQL database using pure SQLAlchemy ORM.
    Contains the mapping layer that extracts dataset/run context from run_info
    and handles ORM relationships.

    Features:
    - Duplicate detection via database constraints (catches IntegrityError)
    - Natural key deduplication (content_hash for Query/Document, fingerprint for Dataset)
    - Uses session_context() for transaction management
    - Logs write operations directly

    Database URL is read from DATABASE_URL environment variable (required).
    Example: postgresql://user:password@localhost:5432/llm_ensemble
    """

    def __init__(self, database_url: str | None = None):
        """Initialize SQL writer with its own logger."""
        self.database_url = database_url
        self.engine = get_engine(database_url)
        self.logger = get_logger(component="sql_writer")

    def write(
        self,
        normalized_dataset: NormalizedDataset,
        run_info: IngestRunInfo,
    ) -> WriteSummary:
        """Write normalized dataset to SQL database with direct logging.

        Duplicate detection via database constraint violations (IntegrityError).
        Tracks created vs skipped entities in WriteSummary.
        Logs each entity type write and summary.

        Args:
            normalized_dataset: Complete normalized dataset with fingerprint and samples
            run_info: Immutable runtime context

        Returns:
            WriteSummary as pure data (metadata for run summary)

        Raises:
            IOError: If database write fails
        """
        dataset_samples = normalized_dataset.samples
        judging_samples = [ds.judging_sample for ds in dataset_samples]

        if not dataset_samples:
            return WriteSummary()

        # Note: Tables must be created via `make db-init` before first write
        # Create summary builder
        summary = WriteSummary()

        # Write to database in transaction
        try:
            with session_context(self.engine) as session:
                # Save entities in strict dependency order to satisfy foreign key constraints
                # Order matters! Each step depends on previous steps being persisted.
                #
                # Dependency graph:
                #   1. Query, Document (no dependencies - global entities)
                #   2. JudgingSample (depends on Query, Document)
                #   3. NormalizedDataset entity (no FK dependencies - just id + fingerprint)
                #   4. NormalizedDataset junction (depends on NormalizedDataset + JudgingSample)
                #   5. IngestRun (depends on NormalizedDataset)

                # Collect unique queries and documents from batch
                unique_queries, unique_documents = self._collect_unique_entities(dataset_samples)

                # 1. Queries (no dependencies - global entities)
                created, skipped = self._save_queries(session, unique_queries)
                summary.add_queries(created=created, skipped=skipped)
                if created > 0 or skipped > 0:
                    self.logger.info(IngestWriteEvent.WRITE_QUERIES, created=created, skipped=skipped)

                # 2. Documents (no dependencies - global entities)
                created, skipped = self._save_documents(session, unique_documents)
                summary.add_documents(created=created, skipped=skipped)
                if created > 0 or skipped > 0:
                    self.logger.info(IngestWriteEvent.WRITE_DOCUMENTS, created=created, skipped=skipped)

                # 3. JudgingSamples (depend on Query + Document)
                created, skipped = self._save_samples(session, judging_samples)
                summary.add_samples(created=created, skipped=skipped)
                if created > 0 or skipped > 0:
                    self.logger.info(IngestWriteEvent.WRITE_JUDGING_SAMPLES, created=created, skipped=skipped)

                # 4. NormalizedDataset entity (no FK dependencies)
                created, skipped = self._save_normalized_dataset_entity(session, normalized_dataset)
                if created > 0 or skipped > 0:
                    self.logger.info(IngestWriteEvent.WRITE_NORMALIZED_DATASET, created=created, skipped=skipped)

                # Flush to ensure NormalizedDataset is persisted before creating DatasetSample records
                # (required because DatasetSample has FK to NormalizedDataset)
                session.flush()

                # 5. DatasetSample records (depend on NormalizedDataset + JudgingSample)
                created, skipped = self._save_dataset_samples(session, normalized_dataset)
                if created > 0 or skipped > 0:
                    self.logger.info(IngestWriteEvent.WRITE_DATASET_SAMPLES, created=created, skipped=skipped)

                # 6. IngestRun (depends on NormalizedDataset)
                created, skipped = self._save_ingest_run(session, run_info, normalized_dataset.id)
                summary.add_runs(created=created, skipped=skipped)
                if created > 0 or skipped > 0:
                    self.logger.info(IngestWriteEvent.WRITE_RUNS, created=created, skipped=skipped)

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

    def _save_ingest_run(
        self, session: Session, run_info: IngestRunInfo, normalized_dataset_id: UUID
    ) -> Tuple[int, int]:
        """Save ingest run entity to database using mapper.

        Uses constraint-based duplicate detection via IntegrityError on run_name.

        Args:
            session: SQLAlchemy session
            run_info: IngestRunInfo context object
            normalized_dataset_id: UUID of the NormalizedDataset this run produced

        Returns:
            Tuple of (created_count, skipped_count)
        """
        try:
            savepoint = session.begin_nested()
            ingest_run_orm = ingest_run_info_to_orm(run_info, normalized_dataset_id)
            session.add(ingest_run_orm)
            session.flush()
            return (1, 0)
        except IntegrityError:
            savepoint.rollback()
            return (0, 1)

    def _collect_unique_entities(
        self, samples: List[DatasetSample]
    ) -> Tuple[Dict[UUID, Query], Dict[UUID, Document]]:
        """Collect unique queries and documents from dataset samples batch."""
        unique_queries: Dict[UUID, Query] = {}
        unique_documents: Dict[UUID, Document] = {}

        for sample in samples:
            judging_sample = sample.judging_sample
            if judging_sample.query.id not in unique_queries:
                unique_queries[judging_sample.query.id] = judging_sample.query
            if judging_sample.document.id not in unique_documents:
                unique_documents[judging_sample.document.id] = judging_sample.document

        return unique_queries, unique_documents

    def _save_queries(
        self, session: Session, queries: Dict[UUID, Query]
    ) -> Tuple[int, int]:
        """Save query entities to database using mapper.

        Uses constraint-based duplicate detection via IntegrityError on content_hash.

        Args:
            session: SQLAlchemy session
            queries: Dictionary of Query domain objects keyed by ID

        Returns:
            Tuple of (created_count, skipped_count)
        """
        created = 0
        skipped = 0

        for query in queries.values():
            try:
                savepoint = session.begin_nested()
                query_orm = query_to_orm(query)
                session.add(query_orm)
                session.flush()
                created += 1
            except IntegrityError:
                savepoint.rollback()
                skipped += 1

        return (created, skipped)

    def _save_documents(
        self, session: Session, documents: Dict[UUID, Document]
    ) -> Tuple[int, int]:
        """Save document entities to database using mapper.

        Uses constraint-based duplicate detection via IntegrityError on content_hash.

        Args:
            session: SQLAlchemy session
            documents: Dictionary of Document domain objects keyed by ID

        Returns:
            Tuple of (created_count, skipped_count)
        """
        created = 0
        skipped = 0

        for document in documents.values():
            try:
                savepoint = session.begin_nested()
                doc_orm = document_to_orm(document)
                session.add(doc_orm)
                session.flush()
                created += 1
            except IntegrityError:
                savepoint.rollback()
                skipped += 1

        return (created, skipped)

    def _save_samples(
        self, session: Session, samples: List[JudgingSample]
    ) -> Tuple[int, int]:
        """Save judging sample entities to database.

        Uses constraint-based duplicate detection via IntegrityError on (query_id, document_id).

        Args:
            session: SQLAlchemy session
            samples: List of JudgingSample domain objects

        Returns:
            Tuple of (created_count, skipped_count)
        """
        created = 0
        skipped = 0

        for sample in samples:
            try:
                savepoint = session.begin_nested()
                sample_orm = judging_sample_to_orm(sample)
                session.add(sample_orm)
                session.flush()
                created += 1
            except IntegrityError:
                savepoint.rollback()
                skipped += 1

        return (created, skipped)

    def _save_normalized_dataset_entity(
        self, session: Session, normalized_dataset: NormalizedDataset
    ) -> Tuple[int, int]:
        """Save NormalizedDataset entity (step 4 in dependency order).

        Uses constraint-based duplicate detection via IntegrityError on fingerprint.

        Note: This MUST be called before _save_dataset_samples because
        DatasetSample records have FK to NormalizedDataset.

        Args:
            session: SQLAlchemy session
            normalized_dataset: NormalizedDataset domain object

        Returns:
            Tuple of (created_count, skipped_count)
        """
        try:
            savepoint = session.begin_nested()
            normalized_dataset_orm = normalized_dataset_to_orm(normalized_dataset)
            session.add(normalized_dataset_orm)
            session.flush()
            return (1, 0)
        except IntegrityError:
            savepoint.rollback()
            return (0, 1)

    def _save_dataset_samples(
        self, session: Session, normalized_dataset: NormalizedDataset
    ) -> Tuple[int, int]:
        """Save DatasetSample records (step 5 in dependency order).

        Uses constraint-based duplicate detection via IntegrityError on
        (normalized_dataset_id, judging_sample_id).

        Note: This MUST be called after _save_normalized_dataset_entity because
        DatasetSample has FK to NormalizedDataset.

        Args:
            session: SQLAlchemy session
            normalized_dataset: NormalizedDataset domain object

        Returns:
            Tuple of (created_count, skipped_count)
        """
        created = 0
        skipped = 0

        # Create DatasetSample records with sequence numbers
        for sample in normalized_dataset.samples:
            try:
                savepoint = session.begin_nested()
                dataset_sample = DatasetSampleORM(
                    id=sample.id,
                    normalized_dataset_id=sample.normalized_dataset_id,
                    judging_sample_id=sample.judging_sample.id,
                    sequence_number=sample.sequence_number,
                )
                session.add(dataset_sample)
                session.flush()
                created += 1
            except IntegrityError:
                savepoint.rollback()
                skipped += 1

        return (created, skipped)
