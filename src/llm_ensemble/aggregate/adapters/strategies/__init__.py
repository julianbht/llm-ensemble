"""Aggregation strategy adapters - concrete implementations of AggregationStrategyPort.

All strategy adapters must be imported here to register with AggregationStrategyRegistry.
This module serves as the single source of truth for available strategies.
"""

from llm_ensemble.aggregate.adapters.strategies.majority_vote_adapter import (
    MajorityVoteAdapter,
)

__all__ = ["MajorityVoteAdapter", "ensure_strategies_registered"]


def ensure_strategies_registered() -> None:
    """Explicit function to ensure all strategies are registered.
    
    Called by CLI and param types to trigger registration.
    Simply referencing the imported classes ensures decorators have run.
    """
    # Force reference to trigger decorator execution
    _ = MajorityVoteAdapter

