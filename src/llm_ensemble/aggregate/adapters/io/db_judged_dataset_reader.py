"""Database adapter for reading InferRunOutputs from infer runs.

Reads InferRunOutput records from PostgreSQL database by infer run name(s).
This adapter queries the infer schema and loads complete InferRunOutput domain
objects for use in the aggregation pipeline.
"""

from __future__ import annotations

from llm_ensemble.infer.schemas.entities.infer_run_output import InferRunOutput
from llm_ensemble.infer.schemas.entities.llm_judgement import LLMJudgement
from llm_ensemble.infer.schemas.orms_normalized import (
    InferRunORM,
    LLMJudgementORM,
)
from llm_ensemble.aggregate.ports import JudgementReader
from llm_ensemble.libs.db import get_engine, get_session


class DbInferRunOutputReader(JudgementReader):
    """Read InferRunOutput records from database by infer run name(s).

    This adapter implements the JudgementReader port by loading InferRunOutputs
    from the infer schema. It queries via InferRun → InferRunOutput relationship
    and reconstructs complete domain objects.

    Database connection:
    - Reads DATABASE_URL from environment (.env file)
    - Uses SQLAlchemy session with eager loading
    - Session closed automatically after read

    Query strategy:
    - For each run_name, find InferRunORM
    - Load linked InferRunOutputORM via FK
    - Query LLMJudgementORM records directly
    - Eager load related entities (prompt, score, metrics)
    - Reconstruct Pydantic InferRunOutput models

    Note: This reader does NOT validate fingerprints or completeness.
    Validation belongs in the aggregation service layer.
    """

    def read(self, run_names: list[str]) -> list[InferRunOutput]:
        """Read InferRunOutput from database by infer run name(s).

        Loads one InferRunOutput per run, each containing the sample_fingerprint
        and all LLM judgements from that run.

        Args:
            run_names: List of infer run identifiers (e.g., ["run1", "run2"])

        Returns:
            List of InferRunOutput objects, one per run

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
                        f"Infer run '{run_name}' has no linked InferRunOutput. "
                        f"This indicates the run did not complete successfully."
                    )

                # TODO: Query and convert LLM judgements from ORM to domain models
                # For now, create empty judgements list until we implement proper ORM-to-domain mappers
                # Will need to query LLMJudgementORM and eagerly load related entities:
                #   - llm_prompt_text + prompt_template + dataset_sample
                #   - llm_score + parser_spec + llm_response_text
                #   - llm_invocation_metrics
                llm_judgements = []

                # Create InferRunOutput domain object
                judged_dataset = InferRunOutput(
                    id=judged_dataset_orm.id,
                    model_config_id=judged_dataset_orm.model_config_id,
                    sample_fingerprint=judged_dataset_orm.sample_fingerprint or "",
                    llm_judgements=llm_judgements,
                )

                judged_datasets.append(judged_dataset)

            return judged_datasets

        finally:
            session.close()
