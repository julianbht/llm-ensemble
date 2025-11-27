"""Schemas for the aggregate CLI."""

from llm_ensemble.aggregate.schemas.aggregated_vote import (
    AggregatedVote,
)
from llm_ensemble.aggregate.schemas.dataset_vote import (
    DatasetVote,
)
from llm_ensemble.aggregate.schemas.aggregated_dataset import (
    AggregatedDataset,
)
from llm_ensemble.aggregate.schemas.aggregated_score import (
    AggregatedScore,
)
from llm_ensemble.aggregate.schemas.aggregated_judgement import (
    AggregatedJudgement,
)

__all__ = [
    "AggregatedVote",
    "DatasetVote",
    "AggregatedDataset",
    "AggregatedScore",
    "AggregatedJudgement",
]
