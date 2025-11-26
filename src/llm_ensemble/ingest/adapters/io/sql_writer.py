"""SQL writer adapter for persisting judging samples to database.

Uses pure SQLAlchemy ORM models with deterministic UUIDs.
Handles its own logging and returns write summary as metadata.

This adapter delegates ORM mapping to the mappers module for bidirectional symmetry.
"""

from __future__ import annotations
from typing import List, Dict, Tuple
from uuid import UUID

from sqlalchemy.orm import Session

from llm_ensemble.ingest.schemas import (
    JudgingSample,
    Query,
    Document,
    WriteSummary,
    NormalizedDataset,
)
from llm_ensemble.ingest.schemas.dataset_sample import DatasetSample
from llm_ensemble.ingest.schemas.ingest_run_info import IngestRunInfo
from llm_ensemble.ingest.schemas.orms import (
    QueryORM,
    DocumentORM,
    IngestRunORM,
    JudgingSampleORM,
    NormalizedDatasetORM,
    DatasetSampleORM,
)
from llm_ensemble.ingest.ports import DatasetWriter
from llm_ensemble.ingest.adapters.io.mappers import (
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
        self.logger = get_logger(component="sql_writer")

    def write(
        self,
        normalized_dataset: NormalizedDataset,
        run_info: IngestRunInfo,
    ) -> WriteSummary:
        """Write normalized dataset to SQL database with direct logging.

        Idempotent operation - merges entities (insert if new, update if exists).
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
                    self.logger.info("write_normalized_dataset", created=created, skipped=skipped)

                # Flush to ensure NormalizedDataset is persisted before creating junction records
                # (required because step 5 has FK to NormalizedDataset)
                session.flush()

                # 5. NormalizedDataset junction records (depend on NormalizedDataset + JudgingSample)
                created = self._save_normalized_dataset_junction(session, normalized_dataset)
                if created > 0:
                    self.logger.info("write_normalized_dataset_junction", created=created)

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

        Args:
            session: SQLAlchemy session
            run_info: IngestRunInfo context object
            normalized_dataset_id: UUID of the NormalizedDataset this run produced

        Returns:
            Tuple of (created_count, skipped_count)
        """
        existing = session.get(IngestRunORM, run_info.id)
        if existing:
            return (0, 1)

        ingest_run_orm = ingest_run_info_to_orm(run_info, normalized_dataset_id)
        session.add(ingest_run_orm)
        return (1, 0)

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

        Args:
            session: SQLAlchemy session
            queries: Dictionary of Query domain objects keyed by ID

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

            query_orm = query_to_orm(query)
            session.merge(query_orm)
            created += 1

        return (created, skipped)

    def _save_documents(
        self, session: Session, documents: Dict[UUID, Document]
    ) -> Tuple[int, int]:
        """Save document entities to database using mapper.

        Args:
            session: SQLAlchemy session
            documents: Dictionary of Document domain objects keyed by ID

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

            doc_orm = document_to_orm(document)
            session.merge(doc_orm)
            created += 1

        return (created, skipped)

    def _save_samples(
        self, session: Session, samples: List[JudgingSample]
    ) -> Tuple[int, int]:
        """Save judging sample entities to database.

        Idempotent operation - reuses existing samples with same ID.

        Args:
            session: SQLAlchemy session
            samples: List of JudgingSample domain objects

        Returns:
            Tuple of (created_count, skipped_count)
        """
        created = 0
        skipped = 0
        seen_ids = set()

        for sample in samples:
            # Duplicates within the same batch won't be visible to session.get()
            if sample.id in seen_ids:
                skipped += 1
                continue
            seen_ids.add(sample.id)

            existing = session.get(JudgingSampleORM, sample.id)
            if not existing:
                sample_orm = judging_sample_to_orm(sample)
                session.add(sample_orm)
                created += 1
            else:
                skipped += 1

        return (created, skipped)

    def _save_normalized_dataset_entity(
        self, session: Session, normalized_dataset: NormalizedDataset
    ) -> Tuple[int, int]:
        """Save NormalizedDataset entity (step 5 in dependency order).

        Idempotent operation - reuses existing NormalizedDataset with same fingerprint.

        Note: This MUST be called before _save_normalized_dataset_junction because
        junction records have FK to NormalizedDataset.

        Args:
            session: SQLAlchemy session
            normalized_dataset: NormalizedDataset domain object

        Returns:
            Tuple of (created_count, skipped_count)
        """
        # Check if NormalizedDataset already exists (idempotency via fingerprint)
        existing = session.get(NormalizedDatasetORM, normalized_dataset.id)
        if existing:
            return (0, 1)

        # Create NormalizedDataset entity
        normalized_dataset_orm = normalized_dataset_to_orm(normalized_dataset)
        session.add(normalized_dataset_orm)

        return (1, 0)

    def _save_normalized_dataset_junction(
        self, session: Session, normalized_dataset: NormalizedDataset
    ) -> int:
        """Save DatasetSample entity records (step 5 in dependency order).

        Creates DatasetSample entities linking NormalizedDataset to JudgingSamples with
        sequence numbers for deterministic ordering.

        Note: This MUST be called after _save_normalized_dataset_entity because
        DatasetSample entities have FK to NormalizedDataset.

        Args:
            session: SQLAlchemy session
            normalized_dataset: NormalizedDataset domain object

        Returns:
            Number of DatasetSample records created
        """
        # Check if DatasetSample records already exist (idempotency)
        existing_count = (
            session.query(DatasetSampleORM)
            .filter_by(normalized_dataset_id=normalized_dataset.id)
            .count()
        )

        if existing_count > 0:
            return 0

        # Create DatasetSample entities with sequence numbers and computed IDs
        for sample in normalized_dataset.samples:
            dataset_sample = DatasetSampleORM(
                id=sample.id,
                normalized_dataset_id=sample.normalized_dataset_id,
                judging_sample_id=sample.judging_sample.id,
                sequence_number=sample.sequence_number,
            )
            session.add(dataset_sample)

        return len(normalized_dataset.samples)
