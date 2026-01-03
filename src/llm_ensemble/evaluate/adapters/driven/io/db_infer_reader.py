"""SQL database adapter for reading evaluation data from infer runs.

Reads LLMJudgement records from PostgreSQL and normalizes to EvaluationData entity.

Read Strategy:
    1. Find InferRun by run_name
    2. Get InferRunOutput (1:1 with InferRun)
    3. Read all LLMJudgement records
    4. Extract gold_score (ground truth) and llm_score.label (prediction)
    5. Build EvaluationData entity via domain builder
"""

from __future__ import annotations

from sqlalchemy.orm import selectinload

from llm_ensemble.evaluate.application.ports.driven.for_input import ForInput
from llm_ensemble.evaluate.domain.entities.evaluation_data import EvaluationData
from llm_ensemble.evaluate.domain.evaluation_data_builder import build_evaluation_data
from llm_ensemble.libs.db.base import get_engine
from llm_ensemble.libs.db.session import get_session
from llm_ensemble.infer.adapters.driven.io.db.orms import (
    InferRunORM,
    InferRunOutputORM,
    LLMJudgementORM,
)
from llm_ensemble.ingest.adapters.driven.io.db.orms import (
    NormalizedDatasetJudgingSampleORM,
)


class DBInferReader(ForInput):
    """Read evaluation data from infer run in SQL database.

    Normalizes infer run judgements into ground_truth vs predictions format.
    """

    def __init__(self, io_name: str):
        """Initialize DB infer reader.

        Args:
            io_name: I/O configuration name
        """
        self.io_name = io_name

    def read(self, input_run_name: str) -> EvaluationData:
        """Read and normalize evaluation data from infer run.

        Args:
            input_run_name: Infer run name

        Returns:
            EvaluationData entity with validated ground truth and predictions

        Raises:
            ValueError: If run not found or data invalid
        """
        engine = get_engine()

        with get_session(engine) as session:
            # Find InferRun by name
            infer_run = session.query(InferRunORM).filter_by(run_name=input_run_name).first()
            if not infer_run:
                raise ValueError(f"Infer run '{input_run_name}' not found in database")

            # Get InferRunOutput (1:1 with InferRun)
            infer_run_output = (
                session.query(InferRunOutputORM)
                .filter_by(id=infer_run.infer_run_output_id)
                .first()
            )
            if not infer_run_output:
                raise ValueError(f"No output found for infer run '{input_run_name}'")

            # Read all judgements with eager loading of llm_score
            judgements = (
                session.query(LLMJudgementORM)
                .filter_by(infer_run_output_id=infer_run_output.id)
                .options(
                    selectinload(LLMJudgementORM.llm_score),
                )
                .all()
            )

            # Extract ground truth and predictions
            ground_truth = []
            predictions = []

            for judgement in judgements:
                # Load normalized_dataset_judging_sample separately (cross-schema join)
                dataset_sample = (
                    session.query(NormalizedDatasetJudgingSampleORM)
                    .filter_by(id=judgement.normalized_dataset_judging_sample_id)
                    .first()
                )

                # Ground truth from dataset_sample.judging_sample.gold_score
                gold_score = dataset_sample.judging_sample.gold_score
                ground_truth.append(gold_score)

                # Prediction from llm_score.label (None if parse failed)
                llm_label = judgement.llm_score.label if judgement.llm_score else None
                predictions.append(llm_label)

            # Build EvaluationData entity via domain builder (validates business rules)
            return build_evaluation_data(
                ground_truth=ground_truth,
                predictions=predictions,
                run_name=input_run_name,
                run_type="infer",
            )
