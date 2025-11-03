"""SQL writer adapter for persisting judging samples to database.

Uses pure SQLAlchemy ORM models with deterministic UUIDs.
Auto-creates tables on first write and raises explicit errors on duplicates.
"""

from __future__ import annotations
from pathlib import Path
from typing import List

from llm_ensemble.ingest.schemas import JudgingSample
from llm_ensemble.ingest.ports import DatasetWriter
from llm_ensemble.ingest.models import (
    DatasetModel,
    QueryModel,
    DocumentModel,
    IngestRunModel,
    JudgingSampleModel,
)
from llm_ensemble.libs.db import (
    get_engine,
    create_all_tables,
    session_context,
    compute_dataset_uuid,
)


class SqlWriter(DatasetWriter):
    """SQL writer adapter for judging samples.
    
    Writes judging samples to SQL database using pure SQLAlchemy ORM.
    
    Features:
    - Auto-creates tables on first write
    - Deterministic UUIDs for all entities (already in Pydantic schemas)
    - Fail-fast on duplicates (raises explicit error)
    - Uses session_context() for transaction management
    
    Database URL is read from DATABASE_URL environment variable.
    Defaults to sqlite:///artifacts/llm_ensemble.db if not set.
    """
    
    def __init__(self, database_url: str | None = None):
        """Initialize SQL writer.
        
        Args:
            database_url: Database connection URL. If None, reads from DATABASE_URL env var.
        """
        self.database_url = database_url
        self.engine = get_engine(database_url)
    
    def write(self, samples: List[JudgingSample], run_dir: Path) -> None:
        """Write judging samples to SQL database.
        
        Args:
            samples: List of judging samples (Pydantic schemas with id fields set)
            run_dir: Run directory (not used for SQL, but required by interface)
        
        Raises:
            ValueError: If duplicate entities are found
            IOError: If database write fails
        """
        if not samples:
            return
        
        # Auto-create tables on first write
        create_all_tables(self.engine)
        
        # Write to database in transaction
        with session_context(self.engine) as session:
            # Extract dataset name and run info from first sample
            dataset_name = samples[0].query.dataset  # All samples from same dataset
            run_info = samples[0].run_info
            
            # 1. Check for duplicate dataset
            dataset_uuid = compute_dataset_uuid(dataset_name)
            existing_dataset = session.get(DatasetModel, dataset_uuid)
            if existing_dataset:
                raise ValueError(
                    f"Duplicate dataset found: name={dataset_name}, id={dataset_uuid}"
                )
            
            # Create dataset
            dataset_model = DatasetModel(
                id=dataset_uuid,
                name=dataset_name,
                description=f"Dataset from config: {run_info.io_config_name}",
            )
            session.add(dataset_model)
            
            # 2. Check for duplicate ingest run
            existing_run = session.get(IngestRunModel, run_info.id)
            if existing_run:
                raise ValueError(
                    f"Duplicate ingest run found: run_id={run_info.run_id}, id={run_info.id}"
                )
            
            # Create ingest run
            ingest_run_model = IngestRunModel(
                id=run_info.id,
                run_id=run_info.run_id,
                io_config_name=run_info.io_config_name,
                input_path=run_info.input_path,
                limit=run_info.limit,
                git_sha=run_info.git_sha,
                git_branch=run_info.git_branch,
                git_is_dirty="true" if not run_info.git_clean else "false",
            )
            session.add(ingest_run_model)
            
            # 3. Process each sample
            for sample in samples:
                # Check for existing query
                existing_query = session.get(QueryModel, sample.query.id)
                if existing_query:
                    raise ValueError(
                        f"Duplicate query found: dataset={sample.query.dataset}, "
                        f"external_id={sample.query.external_id}, id={sample.query.id}"
                    )
                
                # Check for existing document
                existing_doc = session.get(DocumentModel, sample.document.id)
                if existing_doc:
                    raise ValueError(
                        f"Duplicate document found: dataset={sample.document.dataset}, "
                        f"external_id={sample.document.external_id}, id={sample.document.id}"
                    )
                
                # Check for existing sample
                existing_sample = session.get(JudgingSampleModel, sample.id)
                if existing_sample:
                    raise ValueError(
                        f"Duplicate judging sample found: dataset={sample.query.dataset}, "
                        f"query={sample.query.external_id}, doc={sample.document.external_id}, "
                        f"id={sample.id}"
                    )
                
                # Create query ORM model
                query_model = QueryModel(
                    id=sample.query.id,
                    dataset_id=dataset_model.id,
                    external_id=sample.query.external_id,
                    query_text=sample.query.query_text,
                )
                session.add(query_model)
                
                # Create document ORM model
                doc_model = DocumentModel(
                    id=sample.document.id,
                    dataset_id=dataset_model.id,
                    external_id=sample.document.external_id,
                    doc_text=sample.document.doc_text,
                )
                session.add(doc_model)
                
                # Create judging sample ORM model
                sample_model = JudgingSampleModel(
                    id=sample.id,
                    dataset_id=dataset_model.id,
                    query_id=sample.query.id,
                    document_id=sample.document.id,
                    ingest_run_id=ingest_run_model.id,
                    gold_score=sample.gold_score,
                )
                session.add(sample_model)
