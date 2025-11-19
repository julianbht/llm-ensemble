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

from sqlalchemy.orm import joinedload

from llm_ensemble.ingest.schemas import JudgingSample
from llm_ensemble.ingest.schemas.orms import (
    JudgingSampleORM,
    QueryORM,
    DocumentORM,
    IngestRunORM,
    NormalizedDatasetORM,
    NormalizedDatasetJudgingSampleORM,
)
from llm_ensemble.ingest.adapters.io.mappers import (
    query_from_orm,
    document_from_orm,
    judging_sample_from_orm,
)
from llm_ensemble.infer.ports import ExampleReader
from llm_ensemble.libs.db import get_engine, get_session


class SqlJudgingSampleReader(ExampleReader):
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
    ) -> list[JudgingSample]:
        """Read judging samples from database by ingest run name.

        Args:
            run_name: Ingest run identifier (e.g., "my_ingest_run")
                     Queries database for samples associated with this run
            limit: Optional maximum number of samples to read

        Returns:
            List of JudgingSample domain objects

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

            # 2. Query judging samples via NormalizedDataset with eager loading
            # Join through NormalizedDataset to get samples in deterministic order
            query = (
                session.query(JudgingSampleORM)
                .join(
                    NormalizedDatasetJudgingSampleORM,
                    NormalizedDatasetJudgingSampleORM.judging_sample_id == JudgingSampleORM.id
                )
                .filter(NormalizedDatasetJudgingSampleORM.normalized_dataset_id == ingest_run.normalized_dataset_id)
                .options(
                    # Eager load query and its dataset
                    joinedload(JudgingSampleORM.query).joinedload(QueryORM.dataset),
                    # Eager load document and its dataset
                    joinedload(JudgingSampleORM.document).joinedload(DocumentORM.dataset),
                )
                .order_by(NormalizedDatasetJudgingSampleORM.sequence_number)
            )

            # Apply limit if specified
            if limit is not None:
                query = query.limit(limit)

            samples_orm = query.all()

            # 3. Convert ORM entities to Pydantic domain models using mappers
            samples = []
            for sample_orm in samples_orm:
                # Reconstruct Query from ORM
                query = query_from_orm(sample_orm.query)

                # Reconstruct Document from ORM
                document = document_from_orm(sample_orm.document)

                # Reconstruct JudgingSample from ORM (with embedded query and document)
                sample = judging_sample_from_orm(sample_orm, query, document)
                samples.append(sample)

            return samples

        finally:
            # Always close session (resource cleanup)
            session.close()
