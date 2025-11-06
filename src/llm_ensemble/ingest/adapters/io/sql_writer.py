"""SQL writer adapter for persisting judging samples to database.

Uses pure SQLAlchemy ORM models with deterministic UUIDs.
Auto-creates tables on first write and returns write summary for transparent logging.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple

from sqlalchemy.orm import Session

from llm_ensemble.ingest.schemas import JudgingSample, Dataset, Query, Document, WriteSummary
from llm_ensemble.ingest.schemas.ingest_run_info import IngestRunInfo
from llm_ensemble.ingest.schemas.orms import (
    DatasetORM,
    QueryORM,
    DocumentORM,
    IngestRunORM,
    JudgingSampleORM,
)
from llm_ensemble.ingest.ports import DatasetWriter
from llm_ensemble.libs.db import (
    get_engine,
    create_all_tables,
    session_context,
)


class SqlWriter(DatasetWriter):
    """SQL writer adapter for judging samples.

    Writes judging samples to SQL database using pure SQLAlchemy ORM.

    Features:
    - Auto-creates tables on first write
    - Deterministic UUIDs for all entities (already in Pydantic schemas)
    - Central shared database across all runs for data accumulation
    - Idempotent writes via merge (insert or update if exists)
    - Uses session_context() for transaction management
    - Returns WriteSummary for transparent logging (separation of concerns)

    Database URL is read from DATABASE_URL environment variable.
    Defaults to sqlite:///artifacts/llm_ensemble.db if not set.
    """

    def __init__(self, database_url: str | None = None):
        """Initialize SQL writer.

        Args:
            database_url: Database connection URL. If None, reads from DATABASE_URL env var
                         or defaults to sqlite:///artifacts/llm_ensemble.db
        """
        self.database_url = database_url
        self.engine = get_engine(database_url)

    def write(self, samples: List[JudgingSample], run_dir: Path) -> WriteSummary:
        """Write judging samples to SQL database.

        Idempotent operation - merges entities (insert if new, update if exists).

        Args:
            samples: List of judging samples (Pydantic schemas with id fields set)
            run_dir: Run directory (not used by SQL writer - writes to centralized database)

        Returns:
            WriteSummary tracking what was created vs. skipped

        Raises:
            IOError: If database write fails
        """
        if not samples:
            return WriteSummary()

        # Auto-create tables on first write
        create_all_tables(self.engine)

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
                # Extract metadata from first sample (all samples from same dataset/run)
                dataset = samples[0].query.dataset
                run_info = samples[0].run_info

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
        """Save dataset entity to database.

        Args:
            session: SQLAlchemy session
            dataset: Dataset Pydantic schema

        Returns:
            Tuple of (created_count, skipped_count)
        """
        existing = session.get(DatasetORM, dataset.id)
        if existing:
            return (0, 1)

        dataset_model = DatasetORM(
            id=dataset.id,
            name=dataset.name,
            description=dataset.description,
        )
        session.add(dataset_model)
        return (1, 0)

    def _save_ingest_run(self, session: Session, run_info: IngestRunInfo) -> Tuple[int, int]:
        """Save ingest run entity to database.

        Args:
            session: SQLAlchemy session
            run_info: IngestRunInfo Pydantic schema

        Returns:
            Tuple of (created_count, skipped_count)
        """
        existing = session.get(IngestRunORM, run_info.id)
        if existing:
            return (0, 1)

        ingest_run_model = IngestRunORM(
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
        session.add(ingest_run_model)
        return (1, 0)

    def _collect_unique_entities(
        self, samples: List[JudgingSample]
    ) -> Tuple[Dict[str, Query], Dict[str, Document]]:
        """Collect unique queries and documents from samples batch.

        Args:
            samples: List of judging samples

        Returns:
            Tuple of (unique_queries_dict, unique_documents_dict) keyed by UUID
        """
        unique_queries = {}
        unique_documents = {}

        for sample in samples:
            if sample.query.id not in unique_queries:
                unique_queries[sample.query.id] = sample.query
            if sample.document.id not in unique_documents:
                unique_documents[sample.document.id] = sample.document

        return unique_queries, unique_documents

    def _save_queries(
        self, session: Session, queries: Dict[str, Query], dataset_id: str
    ) -> Tuple[int, int]:
        """Save query entities to database.

        Args:
            session: SQLAlchemy session
            queries: Dictionary of Query Pydantic schemas keyed by ID
            dataset_id: Parent dataset UUID

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

            query_model = QueryORM(
                id=query.id,
                dataset_id=dataset_id,
                external_id=query.external_id,
                query_text=query.query_text,
            )
            session.merge(query_model)
            created += 1

        return (created, skipped)

    def _save_documents(
        self, session: Session, documents: Dict[str, Document], dataset_id: str
    ) -> Tuple[int, int]:
        """Save document entities to database.

        Args:
            session: SQLAlchemy session
            documents: Dictionary of Document Pydantic schemas keyed by ID
            dataset_id: Parent dataset UUID

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

            doc_model = DocumentORM(
                id=document.id,
                dataset_id=dataset_id,
                external_id=document.external_id,
                doc_text=document.doc_text,
            )
            session.merge(doc_model)
            created += 1

        return (created, skipped)

    def _save_samples(
        self, session: Session, samples: List[JudgingSample], run_info: IngestRunInfo
    ) -> Tuple[int, int]:
        """Save judging sample entities to database.

        Args:
            session: SQLAlchemy session
            samples: List of JudgingSample Pydantic schemas
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

            sample_model = JudgingSampleORM(
                id=sample.id,
                query_id=sample.query.id,
                document_id=sample.document.id,
                ingest_run_id=run_info.id,
                run_name=run_info.run_name,
                gold_score=sample.gold_score,
            )
            session.merge(sample_model)
            created += 1

        return (created, skipped)
