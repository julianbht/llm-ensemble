"""Strategy adapters - concrete implementations of AggregationStrategy port."""

from llm_ensemble.aggregate.adapters.strategies.majority_vote_adapter import (
    MajorityVoteAdapter,
)

__all__ = ["MajorityVoteAdapter"]
