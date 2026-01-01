"""SQL database adapter for reading evaluation data from aggregate runs.

Reads AggregatedVote records from PostgreSQL and normalizes to EvaluationData entity.

Read Strategy:
    1. Find AggregateRun by run_name
    2. Get AggregatedDataset by aggregate_run_id
    3. Read all AggregatedVote records
    4. Extract gold_score (ground truth) and final_label (prediction)
    5. Build EvaluationData entity via domain builder
"""

from __future__ import annotations

from sqlalchemy.orm import selectinload

from llm_ensemble.evaluate.application.ports.driven.for_input import ForInput
from llm_ensemble.evaluate.domain.entities.evaluation_data import EvaluationData
from llm_ensemble.evaluate.domain.evaluation_data_builder import build_evaluation_data
from llm_ensemble.libs.db.base import get_engine
from llm_ensemble.libs.db.session import get_session
from llm_ensemble.aggregate.adapters.driven.io.orms import (
    AggregateRunORM,
    AggregatedDatasetORM,
    AggregatedVoteORM,
)
from llm_ensemble.ingest.adapters.driven.io.db.orms import (
    DatasetSampleORM,
)


class DBAggregateReader(ForInput):
    """Read evaluation data from aggregate run in SQL database.

    Normalizes aggregate run votes into ground_truth vs predictions format.
    """

    def __init__(self, io_name: str):
        """Initialize DB aggregate reader.

        Args:
            io_name: I/O configuration name
        """
        self.io_name = io_name

    def read(self, input_run_name: str) -> EvaluationData:
        """Read and normalize evaluation data from aggregate run.

        Args:
            input_run_name: Aggregate run name

        Returns:
            EvaluationData entity with validated ground truth and predictions

        Raises:
            ValueError: If run not found or data invalid
        """
        engine = get_engine()

        with get_session(engine) as session:
            # Find AggregateRun by name
            aggregate_run = session.query(AggregateRunORM).filter_by(run_name=input_run_name).first()
            if not aggregate_run:
                raise ValueError(f"Aggregate run '{input_run_name}' not found in database")

            # Get AggregatedDataset
            aggregated_dataset = (
                session.query(AggregatedDatasetORM)
                .filter_by(aggregate_run_id=aggregate_run.id)
                .first()
            )
            if not aggregated_dataset:
                raise ValueError(f"No aggregated dataset found for aggregate run '{input_run_name}'")

            # Read all aggregated votes with eager loading
            votes = (
                session.query(AggregatedVoteORM)
                .filter_by(aggregated_dataset_id=aggregated_dataset.id)
                .options(
                    selectinload(AggregatedVoteORM.llm_judgements)
                )
                .all()
            )

            # Extract ground truth and predictions
            ground_truth = []
            predictions = []

            for vote in votes:
                # Validate business rule: all judgements in vote must judge same sample
                if len(vote.llm_judgements) == 0:
                    raise ValueError(f"AggregatedVote {vote.id} has no judgements")

                first_dataset_sample_id = vote.llm_judgements[0].dataset_sample_id
                for judgement in vote.llm_judgements:
                    assert judgement.dataset_sample_id == first_dataset_sample_id, (
                        f"Business rule violation: All judgements in vote {vote.id} "
                        f"must judge the same dataset sample"
                    )

                # Load dataset_sample separately (cross-schema join)
                dataset_sample = (
                    session.query(DatasetSampleORM)
                    .filter_by(id=first_dataset_sample_id)
                    .first()
                )

                # Ground truth from dataset_sample.judging_sample.gold_score
                gold_score = dataset_sample.judging_sample.gold_score
                ground_truth.append(gold_score)

                # Prediction from final_label
                predictions.append(vote.final_label)

            # Build EvaluationData entity via domain builder (validates business rules)
            return build_evaluation_data(
                ground_truth=ground_truth,
                predictions=predictions,
                run_name=input_run_name,
                run_type="aggregate",
            )
