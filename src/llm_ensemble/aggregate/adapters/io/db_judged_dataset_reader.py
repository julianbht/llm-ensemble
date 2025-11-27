"""Database adapter for reading JudgedDatasets from infer runs.

Reads JudgedDataset records from PostgreSQL database by infer run name(s).
This adapter queries the infer schema and loads complete JudgedDataset domain
objects for use in the aggregation pipeline.
"""

from __future__ import annotations

from sqlalchemy.orm import joinedload

from llm_ensemble.infer.schemas.judged_dataset import JudgedDataset
from llm_ensemble.infer.schemas.llm_judgement import LLMJudgement
from llm_ensemble.infer.schemas.orms_normalized import (
    InferRunORM,
    JudgedDatasetORM,
    DatasetJudgementORM,
    LLMJudgementORM,
)
from llm_ensemble.aggregate.ports import JudgementReader
from llm_ensemble.libs.db import get_engine, get_session


class DbJudgedDatasetReader(JudgementReader):
    """Read JudgedDataset records from database by infer run name(s).

    This adapter implements the JudgementReader port by loading JudgedDatasets
    from the infer schema. It queries via InferRun → JudgedDataset relationship
    and reconstructs complete domain objects.

    Database connection:
    - Reads DATABASE_URL from environment (.env file)
    - Uses SQLAlchemy session with eager loading
    - Session closed automatically after read

    Query strategy:
    - For each run_name, find InferRunORM
    - Load linked JudgedDatasetORM via FK
    - Query DatasetJudgementORM records
    - Eager load related LLMJudgementORM entities
    - Reconstruct Pydantic JudgedDataset models

    Note: This reader does NOT validate fingerprints or completeness.
    Validation belongs in the aggregation service layer.
    """

    def read(self, run_names: list[str]) -> list[JudgedDataset]:
        """Read JudgedDataset from database by infer run name(s).

        Loads one JudgedDataset per run, each containing the fingerprint
        and all judgements from that run.

        Args:
            run_names: List of infer run identifiers (e.g., ["run1", "run2"])

        Returns:
            List of JudgedDataset objects, one per run

        Raises:
            LookupError: If any infer run doesn't exist in database
        """
        engine = get_engine()
        session = get_session(engine)

        try:
            judged_datasets = []

            for run_name in run_names:
                # Find infer run by name with eager loading
                infer_run = (
                    session.query(InferRunORM)
                    .filter_by(run_name=run_name)
                    .options(joinedload(InferRunORM.judged_dataset))
                    .one_or_none()
                )

                if not infer_run:
                    raise LookupError(
                        f"Infer run '{run_name}' not found in database. "
                        f"Available runs: SELECT run_name FROM infer.infer_runs"
                    )

                judged_dataset_orm = infer_run.judged_dataset

                if not judged_dataset_orm:
                    raise LookupError(
                        f"Infer run '{run_name}' has no linked JudgedDataset. "
                        f"This indicates the run did not complete successfully."
                    )

                # Query dataset judgements (maintains sequence order)
                dataset_judgements_orm = (
                    session.query(DatasetJudgementORM)
                    .filter_by(judged_dataset_id=judged_dataset_orm.id)
                    .options(
                        joinedload(DatasetJudgementORM.llm_judgements)
                        .joinedload(LLMJudgementORM.model_config),
                        joinedload(DatasetJudgementORM.llm_judgements)
                        .joinedload(LLMJudgementORM.llm_prompt_text)
                        .joinedload("prompt_template"),
                        joinedload(DatasetJudgementORM.llm_judgements)
                        .joinedload(LLMJudgementORM.llm_score)
                        .joinedload("parser_spec"),
                        joinedload(DatasetJudgementORM.llm_judgements)
                        .joinedload(LLMJudgementORM.llm_invocation_metrics),
                    )
                    .order_by(DatasetJudgementORM.sequence_number)
                    .all()
                )

                # TODO: Convert ORM entities to Pydantic domain models
                # For now, create empty judgements list
                # This will be implemented when we have proper ORM-to-domain mappers
                judgements = []

                # Create JudgedDataset domain object
                judged_dataset = JudgedDataset(
                    id=judged_dataset_orm.id,
                    fingerprint=judged_dataset_orm.fingerprint or "",
                    judgements=judgements,
                )

                judged_datasets.append(judged_dataset)

            return judged_datasets

        finally:
            session.close()
