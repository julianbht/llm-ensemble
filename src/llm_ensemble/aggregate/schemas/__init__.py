"""Schemas for the aggregate CLI."""

from llm_ensemble.aggregate.schemas.aggregated_vote import (
    AggregatedVote,
)
from llm_ensemble.aggregate.schemas.aggregated_dataset import (
    AggregatedDataset,
)

__all__ = [
    "AggregatedVote",
    "DatasetVote",
    "AggregatedDataset",
]
