"""SQL writer adapter for persisting judging samples to database.

Uses pure SQLAlchemy ORM models with deterministic UUIDs.
Auto-creates tables on first write and raises explicit errors on duplicates.
"""

from __future__ import annotations
from pathlib import Path
from typing import List

from llm_ensemble.ingest.schemas import JudgingSample
from llm_ensemble.ingest.schemas.orms import (
    DatasetModel,
    QueryModel,
    DocumentModel,
    IngestRunModel,
    JudgingSampleModel,
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

    def write(self, samples: List[JudgingSample], run_dir: Path) -> None:
        """Write judging samples to SQL database.

        Idempotent operation - merges entities (insert if new, update if exists).

        Args:
            samples: List of judging samples (Pydantic schemas with id fields set)
            run_dir: Run directory (not used for SQL, but required by interface)

        Raises:
            IOError: If database write fails
        """
        if not samples:
            return

        # Auto-create tables on first write
        create_all_tables(self.engine)
        
        # Write to database in transaction
        try:
            with session_context(self.engine) as session:
                # Extract dataset name and run info from first sample
                dataset = samples[0].query.dataset  # All samples from same dataset
                run_info = samples[0].run_info

                # 1. Get or skip dataset (ignore if exists)
                dataset_model = session.get(DatasetModel, dataset.id)
                if not dataset_model:
                    dataset_model = DatasetModel(
                        id=dataset.id,
                        name=dataset.name,
                        description=dataset.description,
                    )
                    session.add(dataset_model)

                # 2. Get or skip ingest run (ignore if exists)
                ingest_run_model = session.get(IngestRunModel, run_info.id)
                if not ingest_run_model:
                    ingest_run_model = IngestRunModel(
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

                # 3. Collect unique queries and documents from batch (keyed by ID)
                unique_queries = {}
                unique_documents = {}

                for sample in samples:
                    if sample.query.id not in unique_queries:
                        unique_queries[sample.query.id] = sample.query
                    if sample.document.id not in unique_documents:
                        unique_documents[sample.document.id] = sample.document

                # 4. Merge all unique queries
                for query in unique_queries.values():
                    query_model = QueryModel(
                        id=query.id,
                        dataset_id=dataset_model.id,
                        external_id=query.external_id,
                        query_text=query.query_text,
                    )
                    session.merge(query_model)

                # 5. Merge all unique documents
                for document in unique_documents.values():
                    doc_model = DocumentModel(
                        id=document.id,
                        dataset_id=dataset_model.id,
                        external_id=document.external_id,
                        doc_text=document.doc_text,
                    )
                    session.merge(doc_model)

                # 6. Merge all samples
                for sample in samples:
                    sample_model = JudgingSampleModel(
                        id=sample.id,
                        query_id=sample.query.id,
                        document_id=sample.document.id,
                        ingest_run_id=ingest_run_model.id,
                        run_name=ingest_run_model.run_name,
                        gold_score=sample.gold_score,
                    )
                    session.merge(sample_model)
        except Exception as e:
            raise IOError(f"Failed to write samples to database: {e}") from e
