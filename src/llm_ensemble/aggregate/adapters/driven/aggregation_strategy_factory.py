"""Builder for aggregation strategies.

Simple, explicit mapping of strategy names to adapter classes.
No decorators, no hidden registration - just a clear dictionary.

To add a new strategy:
1. Create adapter class that extends AggregationStrategyPort
2. Import it here
3. Add to STRATEGIES dict
"""

from __future__ import annotations
from typing import Dict, Type

from llm_ensemble.aggregate.application.ports.driven.for_aggregating import ForAggregating
from llm_ensemble.aggregate.adapters.driven.strategies.majority_vote_adapter import MajorityVoteAdapter
from llm_ensemble.aggregate.adapters.driven.strategies.average_vote_adapter import AverageVoteAdapter


# Explicit mapping of strategy names to adapter classes
STRATEGIES: Dict[str, Type[ForAggregating]] = {
    "majority_vote": MajorityVoteAdapter,
    "average_vote": AverageVoteAdapter,
}


class AggregationStrategyBuilder:
    """Builder for creating aggregation strategy instances."""
    
    @staticmethod
    def build(strategy_name: str) -> ForAggregating:
        """Build and return a strategy adapter instance.
        
        Args:
            strategy_name: Name of the strategy (e.g., 'majority_vote')
            
        Returns:
            Instantiated strategy adapter
            
        Raises:
            ValueError: If strategy not found
        """
        if strategy_name not in STRATEGIES:
            available = ", ".join(sorted(STRATEGIES.keys()))
            raise ValueError(
                f"Aggregation strategy '{strategy_name}' not found. "
                f"Available: {available}"
            )
        
        adapter_class = STRATEGIES[strategy_name]
        return adapter_class(strategy_name=strategy_name)
    
    @staticmethod
    def list_available() -> list[str]:
        """List all available strategy names.
        
        Returns:
            Sorted list of strategy names
        """
        return sorted(STRATEGIES.keys())
    
    @staticmethod
    def has_strategy(strategy_name: str) -> bool:
        """Check if strategy is available.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            True if strategy exists
        """
        return strategy_name in STRATEGIES
