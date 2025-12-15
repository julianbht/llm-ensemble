"""SQL database adapter for reading judging samples.

Reads JudgingSample records from PostgreSQL database by ingest run name.
This adapter queries the normalized relational schema and reconstructs
domain objects from ORM entities using the mappers module for symmetry.

Read Strategy:
    Queries samples via NormalizedDataset for deterministic ordering:
    1. Find IngestRun by run_name
    2. Get IngestRun.normalized_dataset_id
    3. Join through NormalizedDatasetJudgingSample junction table
    4. Order by sequence_number (ensures reproducible ordering)
    5. Eager load Query -> Dataset and Document -> Dataset
    6. Reconstruct domain objects with embedded datasets

The adapter follows the same database connection pattern as SqlWriter,
using SQLAlchemy sessions from the libs/db layer.
"""

from __future__ import annotations
from typing import Optional

from llm_ensemble.ingest.schemas.normalized_dataset import NormalizedDataset
from llm_ensemble.ingest.schemas.orms import (
    JudgingSampleORM,
    QueryORM,
    DocumentORM,
    IngestRunORM,
    NormalizedDatasetORM,
    DatasetSampleORM,
)
from llm_ensemble.ingest.adapters.io.mappers import (
    query_from_orm,
    document_from_orm,
    judging_sample_from_orm,
    dataset_sample_from_orm,
    normalized_dataset_from_orm,
)
from llm_ensemble.infer.ports.input_port import InputPort
from llm_ensemble.libs.db import get_engine, get_session


class DBReader(InputPort):
    """Read JudgingSample records from SQL database by ingest run name.

    This adapter implements the ExampleReader port while handling the
    impedance mismatch between relational entities and domain objects.

    Architecture:
    - Implements same interface as FullyPopulatedJsonReader (unified port)
    - Data mapper logic lives inside this adapter (preserves domain purity)
    - Queries by ingest run name (string parameter)

    Database connection:
    - Reads DATABASE_URL from environment (.env file)
    - Uses SQLAlchemy session with eager loading
    - Session closed automatically after read

    Query strategy:
    - Find IngestRunORM by run_name
    - Query JudgingSampleORM records for that run
    - Eager load related Query, Document, Dataset entities (avoid N+1)
    - Reconstruct Pydantic domain models from ORM entities
    """

    def read(
        self,
        run_name: str,
        limit: Optional[int] = None,
    ) -> NormalizedDataset:
        """Read normalized dataset from database by ingest run name.

        Args:
            run_name: Ingest run identifier (e.g., "my_ingest_run")
                     Queries database for samples associated with this run
            limit: Optional maximum number of samples to read

        Returns:
            NormalizedDataset domain object with DatasetSample entities

        Raises:
            FileNotFoundError: If ingest run doesn't exist in database
            ValueError: If database query fails or data is invalid
        """
        # Get database engine and create session
        engine = get_engine()  # Reads DATABASE_URL from .env
        session = get_session(engine)

        try:
            # 1. Find ingest run by name
            ingest_run = (
                session.query(IngestRunORM)
                .filter_by(run_name=run_name)
                .one_or_none()
            )

            if not ingest_run:
                raise ValueError(
                    f"Ingest run '{run_name}' not found in database."
                )

            # 2. Query DatasetSample entities via NormalizedDataset
            # Order by sequence_number for deterministic ordering
            query = (
                session.query(DatasetSampleORM)
                .filter(DatasetSampleORM.normalized_dataset_id == ingest_run.normalized_dataset_id)
                .order_by(DatasetSampleORM.sequence_number)
            )

            # Apply limit if specified
            if limit is not None:
                query = query.limit(limit)

            dataset_sample_orms = query.all()

            # 3. Get the NormalizedDatasetORM for metadata
            normalized_dataset_orm = (
                session.query(NormalizedDatasetORM)
                .filter_by(id=ingest_run.normalized_dataset_id)
                .one()
            )

            # 4. Convert ORM entities to Pydantic domain models using mappers
            dataset_samples = []
            for ds_orm in dataset_sample_orms:
                # Manually fetch related JudgingSample
                judging_sample_orm = session.get(JudgingSampleORM, ds_orm.judging_sample_id)
                if not judging_sample_orm:
                    raise ValueError(f"JudgingSample {ds_orm.judging_sample_id} not found")

                # Manually fetch related Query and Document
                query_orm = session.get(QueryORM, judging_sample_orm.query_id)
                document_orm = session.get(DocumentORM, judging_sample_orm.document_id)

                if not query_orm or not document_orm:
                    raise ValueError("Query or Document not found for JudgingSample")

                # Reconstruct Query from ORM
                query_obj = query_from_orm(query_orm)

                # Reconstruct Document from ORM
                document = document_from_orm(document_orm)

                # Reconstruct JudgingSample from ORM (with embedded query and document)
                judging_sample = judging_sample_from_orm(judging_sample_orm, query_obj, document)

                # Reconstruct DatasetSample from ORM (with embedded judging_sample)
                dataset_sample = dataset_sample_from_orm(ds_orm, judging_sample)
                dataset_samples.append(dataset_sample)

            # 5. Reconstruct NormalizedDataset from ORM with DatasetSamples
            normalized_dataset = normalized_dataset_from_orm(
                normalized_dataset_orm,
                dataset_samples
            )

            return normalized_dataset

        finally:
            # Always close session (resource cleanup)
            session.close()
