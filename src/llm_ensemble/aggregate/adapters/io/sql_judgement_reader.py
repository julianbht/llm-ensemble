"""SQL database adapter for reading LLM judgements from infer runs.

Reads JudgedDataset records from PostgreSQL database by infer run name(s).
This adapter queries the normalized relational schema via InferRun → JudgedDataset
relationship and reconstructs complete domain objects from ORM entities.

Includes validation that all runs processed the same samples (same fingerprint).

The adapter follows the same database connection pattern as SqlJudgementWriter,
using SQLAlchemy sessions from the libs/db layer.
"""

from __future__ import annotations

from sqlalchemy.orm import joinedload

from llm_ensemble.infer.schemas.llm_judgement import LLMJudgement
from llm_ensemble.infer.schemas.judged_dataset import JudgedDataset
from llm_ensemble.infer.schemas.orms_normalized import (
    LLMCallORM,
    LLMRequestORM,
    LLMScoreORM,
    InferRunORM,
    JudgedDatasetORM,
    JudgedDatasetLLMCallORM,
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
    """Read JudgedDataset records from SQL database by infer run name(s).

    This adapter implements the JudgementReader port while handling the
    impedance mismatch between relational entities and domain objects.

    Architecture:
    - Queries via InferRun → JudgedDataset relationship (simplified design)
    - Data mapper logic lives in mappers_orm_to_domain module
    - Validates all JudgedDatasets have same fingerprint (aggregation requirement)

    Database connection:
    - Reads DATABASE_URL from environment (.env file)
    - Uses SQLAlchemy session with eager loading
    - Session closed automatically after read

    Query strategy:
    - For each run_name, find InferRunORM
    - Load linked JudgedDatasetORM via FK
    - Query LLMCallORM records via junction table (preserves sequence)
    - Eager load related entities (request, score, judging sample, etc.)
    - Reconstruct Pydantic LLMJudgement and JudgedDataset models
    """

    def read(self, run_names: list[str]) -> list[JudgedDataset]:
        """Read JudgedDataset from database by infer run name(s).

        Loads one JudgedDataset per run, each containing the fingerprint
        and all judgements from that run. Validates that all JudgedDatasets
        have the same fingerprint (ensuring same samples were processed).

        Args:
            run_names: List of infer run identifiers (e.g., ["run1", "run2"])
                      Queries database for judgements from these runs

        Returns:
            List of JudgedDataset objects, one per run

        Raises:
            LookupError: If any infer run doesn't exist in database
            ValueError: If JudgedDataset fingerprints don't match across runs
        """
        # Get database engine and create session
        engine = get_engine()  # Reads DATABASE_URL from .env
        session = get_session(engine)

        try:
            judged_datasets = []
            fingerprints_seen = set()

            for run_name in run_names:
                # Find infer run by name with eager loading of judged_dataset
                infer_run = (
                    session.query(InferRunORM)
                    .filter_by(run_name=run_name)
                    .options(joinedload(InferRunORM.judged_dataset))
                    .one_or_none()
                )

                if not infer_run:
                    raise LookupError(
                        f"Infer run '{run_name}' not found in database. "
                        f"Available runs can be queried with: SELECT run_name FROM infer.infer_runs"
                    )

                # Check that run completed successfully
                if not infer_run.judged_dataset_id:
                    raise ValueError(
                        f"Infer run '{run_name}' did not complete successfully "
                        f"(judged_dataset_id is NULL). Cannot aggregate incomplete runs."
                    )

                # Get JudgedDataset fingerprint for validation
                judged_dataset_orm = infer_run.judged_dataset
                if not judged_dataset_orm.fingerprint:
                    raise ValueError(
                        f"JudgedDataset for run '{run_name}' has NULL fingerprint. "
                        f"This indicates the run did not complete properly."
                    )

                fingerprints_seen.add(judged_dataset_orm.fingerprint)

                # Query LLM calls via junction table (preserves deterministic ordering)
                # We need to reconstruct full LLMJudgement objects which require:
                # - LLMCall (latency, retries, cost, tokens)
                # - LLMRequest (prompt, judging_sample)
                # - LLMScore (raw_response, parsed fields, parser warnings)
                # - JudgingSample (query, document, gold score)
                calls_orm = (
                    session.query(LLMCallORM)
                    .join(
                        JudgedDatasetLLMCallORM,
                        LLMCallORM.id == JudgedDatasetLLMCallORM.llm_call_id
                    )
                    .filter(
                        JudgedDatasetLLMCallORM.judged_dataset_id == judged_dataset_orm.id
                    )
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
                    .order_by(JudgedDatasetLLMCallORM.sequence_number)
                    .all()
                )

                # Convert ORM entities to Pydantic domain models
                judgements = []
                for call_orm in calls_orm:
                    judgement = llm_judgement_from_orm(call_orm)
                    judgements.append(judgement)

                # Create JudgedDataset domain object
                judged_dataset = JudgedDataset(
                    id=judged_dataset_orm.id,
                    fingerprint=judged_dataset_orm.fingerprint,
                    judgements=judgements,
                )

                judged_datasets.append(judged_dataset)

            # Validate all fingerprints match (aggregation requirement)
            if len(fingerprints_seen) > 1:
                raise ValueError(
                    f"Cannot aggregate runs with different JudgedDataset fingerprints. "
                    f"Found {len(fingerprints_seen)} distinct fingerprints: {fingerprints_seen}. "
                    f"This means the runs processed different sets of samples."
                )

            return judged_datasets

        finally:
            # Always close session (resource cleanup)
            session.close()
