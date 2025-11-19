"""SQL database adapter for reading LLM judgements from infer runs.

Reads LLMJudgement records from PostgreSQL database by infer run name(s).
This adapter queries the normalized relational schema and reconstructs
complete domain objects from ORM entities.

The adapter follows the same database connection pattern as SqlJudgementWriter,
using SQLAlchemy sessions from the libs/db layer.
"""

from __future__ import annotations

from sqlalchemy.orm import joinedload

from llm_ensemble.infer.schemas.llm_judgement import LLMJudgement
from llm_ensemble.infer.schemas.inferred_dataset import InferredDataset
from llm_ensemble.infer.schemas.orms_normalized import (
    LLMCallORM,
    LLMRequestORM,
    LLMScoreORM,
    InferRunORM,
)
from llm_ensemble.ingest.schemas.orms import (
    JudgingSampleORM,
    QueryORM,
    DocumentORM,
)
from llm_ensemble.aggregate.ports import JudgementReader
from llm_ensemble.infer.adapters.io.mappers_orm_to_domain import llm_judgement_from_orm
from llm_ensemble.libs.db import get_engine, get_session


class SqlJudgementReader(JudgementReader):
    """Read LLMJudgement records from SQL database by infer run name(s).

    This adapter implements the JudgementReader port while handling the
    impedance mismatch between relational entities and domain objects.

    Architecture:
    - Implements same interface as JsonJudgementReader (unified port)
    - Data mapper logic lives in sql_mappers module
    - Queries by infer run names (list of string parameters)

    Database connection:
    - Reads DATABASE_URL from environment (.env file)
    - Uses SQLAlchemy session with eager loading
    - Session closed automatically after read

    Query strategy:
    - For each run_name, find InferRunORM
    - Query LLMCallORM records for those runs
    - Eager load related entities (request, response, judging sample, etc.)
    - Reconstruct Pydantic LLMJudgement models from ORM entities
    """

    def read(self, run_names: list[str]) -> list[InferredDataset]:
        """Read InferredDataset from database by infer run name(s).

        Loads one InferredDataset per run, each containing the fingerprint
        and all judgements from that run.

        Args:
            run_names: List of infer run identifiers (e.g., ["run1", "run2"])
                      Queries database for judgements from these runs

        Returns:
            List of InferredDataset objects, one per run

        Raises:
            LookupError: If any infer run doesn't exist in database
        """
        # Get database engine and create session
        engine = get_engine()  # Reads DATABASE_URL from .env
        session = get_session(engine)

        try:
            inferred_datasets = []

            for run_name in run_names:
                # Find infer run by name with eager loading of inferred_dataset
                infer_run = (
                    session.query(InferRunORM)
                    .filter_by(run_name=run_name)
                    .options(joinedload(InferRunORM.inferred_dataset))
                    .one_or_none()
                )

                if not infer_run:
                    raise LookupError(
                        f"Infer run '{run_name}' not found in database. "
                        f"Available runs can be queried with: SELECT run_name FROM infer.infer_runs"
                    )

                # Query LLM calls for this run with comprehensive eager loading
                # We need to reconstruct full LLMJudgement objects which require:
                # - LLMCall (latency, retries, cost, tokens)
                # - LLMRequest (prompt, judging_sample)
                # - LLMScore (raw_response, parsed fields, parser warnings)
                # - JudgingSample (query, document, gold score)
                calls_orm = (
                    session.query(LLMCallORM)
                    .filter_by(infer_run_id=infer_run.id)
                    .options(
                        # Load request and its judging sample with query/document
                        joinedload(LLMCallORM.llm_request)
                        .joinedload(LLMRequestORM.judging_sample_id)
                        .joinedload(JudgingSampleORM.query)
                        .joinedload(QueryORM.dataset),
                        joinedload(LLMCallORM.llm_request)
                        .joinedload(LLMRequestORM.judging_sample_id)
                        .joinedload(JudgingSampleORM.document)
                        .joinedload(DocumentORM.dataset),
                        # Load score with parser spec
                        joinedload(LLMCallORM.score).joinedload(
                            LLMScoreORM.parser_spec
                        ),
                    )
                    .all()
                )

                # Convert ORM entities to Pydantic domain models
                judgements = []
                for call_orm in calls_orm:
                    judgement = llm_judgement_from_orm(call_orm)
                    judgements.append(judgement)

                # Create InferredDataset from judgements
                inferred_dataset = InferredDataset.create(judgements)

                inferred_datasets.append(inferred_dataset)

            return inferred_datasets

        finally:
            # Always close session (resource cleanup)
            session.close()
