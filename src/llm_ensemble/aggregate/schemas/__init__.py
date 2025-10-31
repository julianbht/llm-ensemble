"""Schemas for the aggregate CLI."""

from llm_ensemble.aggregate.schemas.aggregated_score import (
    AggregatedScore,
    PerModelVote,
)
from llm_ensemble.aggregate.schemas.aggregated_judgement import (
    AggregatedJudgement,
)

__all__ = [
    "AggregatedScore",
    "PerModelVote",
    "AggregatedJudgement",
]
