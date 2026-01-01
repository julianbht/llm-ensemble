"""SQL writer adapter for persisting judging samples to database.

Uses pure SQLAlchemy ORM models with random UUIDs.
Duplicate detection via pre-querying existing entities (no exception handling).
Returns UUID mappings to ensure correct foreign key references across runs.
Handles its own logging and returns write summary as metadata.

This adapter delegates ORM mapping to the mappers module for bidirectional symmetry.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple

from sqlalchemy.orm import Session
from sqlalchemy import tuple_

from llm_ensemble.ingest.domain.entities.query import Query
from llm_ensemble.ingest.domain.entities.document import Document
from llm_ensemble.ingest.domain.entities.judging_sample import JudgingSample
from llm_ensemble.ingest.domain.entities.normalized_dataset import NormalizedDataset
from llm_ensemble.ingest.domain.entities.ingest_run_config import IngestRunConfig
from llm_ensemble.ingest.domain.entities.dataset_sample import DatasetSample
from llm_ensemble.ingest.domain.entities.ingest_run import IngestRun
from llm_ensemble.ingest.domain.entities.write_summary import WriteSummary
from llm_ensemble.ingest.adapters.driven.io.db.orms import (
    QueryORM,
    DocumentORM,
    JudgingSampleORM,
    DatasetSampleORM,
    NormalizedDatasetORM,
    IngestRunConfigORM,
    IngestRunORM,
)
from llm_ensemble.ingest.application.ports.driven.for_output import ForOutput
from llm_ensemble.libs.logging.structlog_logger import get_logger
from llm_ensemble.libs.db.base import get_engine
from llm_ensemble.libs.db.session import session_context

from llm_ensemble.libs.logging.log_events import IngestWriteEvent


class DbWriter(ForOutput):
    """SQL writer adapter for judging samples - handles ORM mapping.

    Writes judging samples to SQL database using pure SQLAlchemy ORM.
    Contains the mapping layer that extracts dataset/run context from run_info
    and handles ORM relationships.

    Features:
    - Duplicate detection via pre-querying existing entities by natural keys
    - Bulk insert operations for maximum performance
    - UUID mapping to ensure correct foreign key references across runs
    - Natural key deduplication (content_hash for Query/Document, fingerprint for Dataset)
    - Uses session_context() for transaction management
    - Logs write operations directly

    Database URL is read from DATABASE_URL environment variable (required).
    Example: postgresql://user:password@localhost:5432/llm_ensemble
    """

    def __init__(self, io_name: str, run_dir: Path, database_url: str | None = None):
        """Initialize SQL writer with IO format name and run directory.

        Args:
            io_name: Name of the IO format (e.g., 'llm_judge_ingest')
            run_dir: Run directory path (for consistency with file-based writers)
            database_url: Optional database URL (defaults to DATABASE_URL env var)
        """
        self.io_name = io_name
        self.run_dir = run_dir
        self.database_url = database_url
        self.engine = get_engine(database_url)
        self.logger = get_logger(component=__name__)

    def write(self, ingest_run: IngestRun) -> WriteSummary:
        """Write ingest run results to SQL database with direct logging.

        Duplicate detection via pre-querying existing entities by natural keys.
        Uses bulk insert operations for queries, documents, and judging samples.
        Maintains UUID mappings to ensure correct foreign key references.
        Tracks created vs skipped entities in WriteSummary.
        Logs each entity type write and summary.

        Args:
            ingest_run: Complete IngestRun aggregate containing config, dataset, and metadata

        Returns:
            WriteSummary as pure data (metadata for run summary)

        Raises:
            IOError: If database write fails
        """
        # Extract dataset from aggregate
        normalized_dataset = ingest_run.normalized_dataset
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
                #   3. IngestRunConfig (no dependencies)
                #   4. NormalizedDataset entity (no dependencies)
                #   5. NormalizedDataset junction (depends on NormalizedDataset + JudgingSample)
                #   6. IngestRun (depends on IngestRunConfig + NormalizedDataset)

                # Collect unique queries and documents from batch
                unique_queries, unique_documents = self._collect_unique_entities(dataset_samples)

                # 1. Queries (no dependencies - global entities)
                created, skipped, query_uuid_map = self._save_queries(session, unique_queries)
                summary.add_queries(created=created, skipped=skipped)
                if created > 0 or skipped > 0:
                    self.logger.info(IngestWriteEvent.WRITE_QUERIES, created=created, skipped=skipped)

                # 2. Documents (no dependencies - global entities)
                created, skipped, document_uuid_map = self._save_documents(session, unique_documents)
                summary.add_documents(created=created, skipped=skipped)
                if created > 0 or skipped > 0:
                    self.logger.info(IngestWriteEvent.WRITE_DOCUMENTS, created=created, skipped=skipped)

                # 3. JudgingSamples (depend on Query + Document)
                # Use UUID mappings to ensure correct foreign key references
                created, skipped, sample_uuid_map = self._save_samples(
                    session, judging_samples, query_uuid_map, document_uuid_map
                )
                summary.add_samples(created=created, skipped=skipped)
                if created > 0 or skipped > 0:
                    self.logger.info(IngestWriteEvent.WRITE_JUDGING_SAMPLES, created=created, skipped=skipped)

                # 4. IngestRunConfig (no FK dependencies)
                created, skipped, config_uuid = self._save_ingest_run_config(session, ingest_run.ingest_run_config)
                summary.add_configs(created=created, skipped=skipped)
                if created > 0 or skipped > 0:
                    self.logger.info(IngestWriteEvent.WRITE_RUN_CONFIG, created=created, skipped=skipped)

                # 5. NormalizedDataset entity (no dependencies)
                created, skipped, dataset_uuid = self._save_normalized_dataset_entity(session, normalized_dataset)
                summary.add_datasets(created=created, skipped=skipped)
                if created > 0 or skipped > 0:
                    self.logger.info(IngestWriteEvent.WRITE_NORMALIZED_DATASET, created=created, skipped=skipped)

                # Flush to ensure NormalizedDataset is persisted before creating DatasetSample records
                # (required because DatasetSample has FK to NormalizedDataset)
                session.flush()

                # 6. DatasetSample records (depend on NormalizedDataset + JudgingSample)
                created, skipped = self._save_dataset_samples(
                    session, normalized_dataset, query_uuid_map, document_uuid_map, sample_uuid_map, dataset_uuid
                )
                if created > 0 or skipped > 0:
                    self.logger.info(IngestWriteEvent.WRITE_DATASET_SAMPLES, created=created, skipped=skipped)

                # 7. IngestRun (depends on IngestRunConfig + NormalizedDataset)
                # Use UUID mappings to ensure correct foreign key references
                created, skipped = self._save_ingest_run(session, ingest_run, config_uuid, dataset_uuid)
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
        
    def _collect_unique_entities(
        self, samples: List[DatasetSample]
    ) -> Tuple[Dict[str, Query], Dict[str, Document]]:
        """Collect unique queries and documents from dataset samples batch.

        Deduplicates by content_hash (natural key) to reduce payload size.
        Database will handle cross-batch deduplication via ON CONFLICT.
        """
        unique_queries: Dict[str, Query] = {}
        unique_documents: Dict[str, Document] = {}

        for sample in samples:
            judging_sample = sample.judging_sample
            query = judging_sample.query
            document = judging_sample.document

            # Deduplicate by natural key (content_hash)
            if query.content_hash not in unique_queries:
                unique_queries[query.content_hash] = query
            if document.content_hash not in unique_documents:
                unique_documents[document.content_hash] = document

        return unique_queries, unique_documents


    def _save_ingest_run(
        self,
        session: Session,
        ingest_run: IngestRun,
        config_uuid: str,
        dataset_uuid: str,
    ) -> Tuple[int, int]:
        """Save ingest run entity to database using pre-query pattern.

        Checks if run_name already exists before inserting.
        Uses provided config and dataset UUIDs for correct foreign key references.

        Args:
            session: SQLAlchemy session
            ingest_run: IngestRun aggregate (contains config and dataset)
            config_uuid: UUID of the IngestRunConfig in database
            dataset_uuid: UUID of the NormalizedDataset in database

        Returns:
            Tuple of (created_count, skipped_count)
        """
        # Check if this run_name already exists
        existing = session.query(IngestRunORM).filter_by(run_name=ingest_run.run_name).first()

        if existing:
            return (0, 1)

        # Insert new run with correct foreign key UUIDs
        ingest_run_orm = IngestRunORM(
            id=ingest_run.id,
            run_name=ingest_run.run_name,
            run_type=ingest_run.run_type,
            ingest_run_config_id=config_uuid,
            normalized_dataset_id=dataset_uuid,
            start_time=ingest_run.start_time,
            end_time=ingest_run.end_time,
            git_sha=ingest_run.git_info.git_sha,
            git_branch=ingest_run.git_info.git_branch,
            git_is_dirty="true" if not ingest_run.git_info.git_clean else "false",
            notes=ingest_run.notes,
        )
        session.add(ingest_run_orm)
        session.flush()
        return (1, 0)

    def _save_ingest_run_config(
        self, session: Session, run_config: IngestRunConfig
    ) -> Tuple[int, int, str]:
        """Save ingest run config entity to database using pre-query pattern.

        Checks if natural key (io_config_name, input_path, limit) already exists.
        Returns UUID of existing or newly created config.

        Args:
            session: SQLAlchemy session
            run_config: IngestRunConfig domain object

        Returns:
            Tuple of (created_count, skipped_count, config_uuid)
        """
        # Check if this config already exists (by natural key)
        existing = (
            session.query(IngestRunConfigORM)
            .filter_by(
                io_config_name=run_config.io_config_name,
                input_path=run_config.input_path,
                limit=run_config.limit,
            )
            .first()
        )

        if existing:
            return (0, 1, str(existing.id))

        # Insert new config
        config_orm = IngestRunConfigORM(
            id=run_config.id,
            io_config_name=run_config.io_config_name,
            input_path=run_config.input_path,
            limit=run_config.limit,
        )
        session.add(config_orm)
        session.flush()
        return (1, 0, str(config_orm.id))

    def _save_queries(
        self, session: Session, queries: Dict[str, Query]
    ) -> Tuple[int, int, Dict[str, str]]:
        """Save query entities to database using bulk insert with pre-filtering.

        Queries existing queries by content_hash, filters to new ones, then bulk inserts.
        Returns mapping of content_hash -> UUID for both existing and new queries.

        Args:
            session: SQLAlchemy session
            queries: Dictionary of Query domain objects keyed by content_hash

        Returns:
            Tuple of (created_count, skipped_count, content_hash_to_uuid_mapping)
        """
        if not queries:
            return (0, 0, {})

        # Convert all queries to ORM objects
        query_orms = [
            QueryORM(
                id=q.id,
                query_text=q.query_text,
                content_hash=q.content_hash,
            )
            for q in queries.values()
        ]
        content_hashes = [q.content_hash for q in queries.values()]

        # Query existing queries and get their UUIDs
        existing_queries = {
            content_hash: str(uuid) for (content_hash, uuid) in
            session.query(QueryORM.content_hash, QueryORM.id)
            .filter(QueryORM.content_hash.in_(content_hashes))
        }

        # Filter to only new queries
        new_query_orms = [
            orm for orm in query_orms
            if orm.content_hash not in existing_queries
        ]

        # Bulk insert new queries
        if new_query_orms:
            session.add_all(new_query_orms)
            session.flush()

        # Build complete mapping: content_hash -> UUID (existing + new)
        content_hash_to_uuid = existing_queries.copy()
        for orm in new_query_orms:
            content_hash_to_uuid[orm.content_hash] = str(orm.id)

        created = len(new_query_orms)
        skipped = len(query_orms) - created

        return (created, skipped, content_hash_to_uuid)

    def _save_documents(
        self, session: Session, documents: Dict[str, Document]
    ) -> Tuple[int, int, Dict[str, str]]:
        """Save document entities to database using bulk insert with pre-filtering.

        Queries existing documents by content_hash, filters to new ones, then bulk inserts.
        Returns mapping of content_hash -> UUID for both existing and new documents.

        Args:
            session: SQLAlchemy session
            documents: Dictionary of Document domain objects keyed by content_hash

        Returns:
            Tuple of (created_count, skipped_count, content_hash_to_uuid_mapping)
        """
        if not documents:
            return (0, 0, {})

        # Convert all documents to ORM objects
        doc_orms = [
            DocumentORM(
                id=d.id,
                doc_text=d.doc_text,
                content_hash=d.content_hash,
            )
            for d in documents.values()
        ]
        content_hashes = [d.content_hash for d in documents.values()]

        # Query existing documents and get their UUIDs
        existing_documents = {
            content_hash: str(uuid) for (content_hash, uuid) in
            session.query(DocumentORM.content_hash, DocumentORM.id)
            .filter(DocumentORM.content_hash.in_(content_hashes))
        }

        # Filter to only new documents
        new_doc_orms = [
            orm for orm in doc_orms
            if orm.content_hash not in existing_documents
        ]

        # Bulk insert new documents
        if new_doc_orms:
            session.add_all(new_doc_orms)
            session.flush()

        # Build complete mapping: content_hash -> UUID (existing + new)
        content_hash_to_uuid = existing_documents.copy()
        for orm in new_doc_orms:
            content_hash_to_uuid[orm.content_hash] = str(orm.id)

        created = len(new_doc_orms)
        skipped = len(doc_orms) - created

        return (created, skipped, content_hash_to_uuid)

    def _save_samples(
        self,
        session: Session,
        samples: List[JudgingSample],
        query_uuid_map: Dict[str, str],
        document_uuid_map: Dict[str, str],
    ) -> Tuple[int, int, Dict[Tuple[str, str], str]]:
        """Save judging sample entities to database using bulk insert with pre-filtering.

        Uses query/document UUID mappings to ensure correct foreign key references.
        Returns mapping of (query_uuid, doc_uuid) -> judging_sample_uuid.

        Args:
            session: SQLAlchemy session
            samples: List of JudgingSample domain objects
            query_uuid_map: Mapping of query content_hash -> actual UUID in DB
            document_uuid_map: Mapping of document content_hash -> actual UUID in DB

        Returns:
            Tuple of (created_count, skipped_count, sample_uuid_mapping)
        """
        if not samples:
            return (0, 0, {})

        # Convert samples to ORM objects using correct UUIDs from mappings
        sample_orms: List[JudgingSampleORM] = []
        for s in samples:
            # Look up the actual UUIDs that exist in the database
            query_uuid = query_uuid_map[s.query.content_hash]
            doc_uuid = document_uuid_map[s.document.content_hash]

            # Create ORM object with correct foreign key references
            orm = JudgingSampleORM(
                id=s.id,
                query_id=query_uuid,
                document_id=doc_uuid,
                gold_score=s.gold_score.value,
            )
            sample_orms.append(orm)

        # Build list of (query_id, document_id) tuples to check
        sample_keys = [(orm.query_id, orm.document_id) for orm in sample_orms]

        # Query existing samples and get their UUIDs
        existing_samples = {
            (str(query_id), str(doc_id)): str(sample_id)
            for (query_id, doc_id, sample_id) in
            session.query(
                JudgingSampleORM.query_id,
                JudgingSampleORM.document_id,
                JudgingSampleORM.id
            )
            .filter(
                tuple_(JudgingSampleORM.query_id, JudgingSampleORM.document_id)
                .in_(sample_keys)
            )
        }

        # Filter to only new samples
        new_sample_orms = [
            orm for orm in sample_orms
            if (orm.query_id, orm.document_id) not in existing_samples
        ]

        # Bulk insert new samples
        if new_sample_orms:
            session.add_all(new_sample_orms)
            session.flush()

        # Build complete mapping: (query_uuid, doc_uuid) -> sample_uuid (existing + new)
        sample_uuid_map = existing_samples.copy()
        for orm in new_sample_orms:
            sample_uuid_map[(str(orm.query_id), str(orm.document_id))] = str(orm.id)

        created = len(new_sample_orms)
        skipped = len(sample_orms) - created

        return (created, skipped, sample_uuid_map)

    def _save_normalized_dataset_entity(
        self, session: Session, normalized_dataset: NormalizedDataset
    ) -> Tuple[int, int, str]:
        """Save NormalizedDataset entity (step 4 in dependency order).

        Checks if fingerprint already exists before inserting.
        Returns UUID of existing or newly created dataset.

        Note: This MUST be called before _save_dataset_samples because
        DatasetSample records have FK to NormalizedDataset.

        Args:
            session: SQLAlchemy session
            normalized_dataset: NormalizedDataset domain object

        Returns:
            Tuple of (created_count, skipped_count, dataset_uuid)
        """
        # Check if this dataset fingerprint already exists
        existing = (
            session.query(NormalizedDatasetORM)
            .filter_by(fingerprint=normalized_dataset.fingerprint)
            .first()
        )

        if existing:
            return (0, 1, str(existing.id))

        # Insert new dataset
        normalized_dataset_orm = NormalizedDatasetORM(
            id=normalized_dataset.id,
            fingerprint=normalized_dataset.fingerprint,
            external_dataset_name=normalized_dataset.external_dataset_name,
        )
        session.add(normalized_dataset_orm)
        session.flush()
        return (1, 0, str(normalized_dataset_orm.id))

    def _save_dataset_samples(
        self,
        session: Session,
        normalized_dataset: NormalizedDataset,
        query_uuid_map: Dict[str, str],
        document_uuid_map: Dict[str, str],
        sample_uuid_map: Dict[Tuple[str, str], str],
        dataset_uuid: str,
    ) -> Tuple[int, int]:
        """Save DatasetSample records using bulk insert with pre-filtering.

        Queries existing DatasetSamples by (dataset_id, sample_id), filters to new ones,
        then bulk inserts. Uses UUID mappings for correct foreign key references.

        Note: This MUST be called after _save_normalized_dataset_entity because
        DatasetSample has FK to NormalizedDataset.

        Args:
            session: SQLAlchemy session
            normalized_dataset: NormalizedDataset domain object
            query_uuid_map: Mapping of query content_hash -> actual UUID in DB
            document_uuid_map: Mapping of document content_hash -> actual UUID in DB
            sample_uuid_map: Mapping of (query_uuid, doc_uuid) -> judging_sample_uuid
            dataset_uuid: UUID of the NormalizedDataset in database

        Returns:
            Tuple of (created_count, skipped_count)
        """
        # Create DatasetSample ORM objects with correct UUIDs
        dataset_sample_orms = []
        for sample in normalized_dataset.samples:
            # Look up the correct judging_sample_id using mappings
            query_uuid = query_uuid_map[sample.judging_sample.query.content_hash]
            doc_uuid = document_uuid_map[sample.judging_sample.document.content_hash]
            judging_sample_uuid = sample_uuid_map[(query_uuid, doc_uuid)]

            dataset_sample = DatasetSampleORM(
                id=sample.id,
                normalized_dataset_id=dataset_uuid,  # Use actual UUID from DB
                judging_sample_id=judging_sample_uuid,
                sequence_number=sample.sequence_number,
            )
            dataset_sample_orms.append(dataset_sample)

        # Query which judging_sample_ids already exist for this dataset
        # Since all samples belong to the same dataset_uuid, we can query efficiently
        existing_sample_ids = {
            str(sample_id) for (sample_id,) in
            session.query(DatasetSampleORM.judging_sample_id)
            .filter(DatasetSampleORM.normalized_dataset_id == dataset_uuid)
        }

        # Filter to only new dataset samples
        new_dataset_sample_orms = [
            orm for orm in dataset_sample_orms
            if orm.judging_sample_id not in existing_sample_ids
        ]

        # Bulk insert new dataset samples
        if new_dataset_sample_orms:
            session.add_all(new_dataset_sample_orms)
            session.flush()

        created = len(new_dataset_sample_orms)
        skipped = len(dataset_sample_orms) - created

        return (created, skipped)
