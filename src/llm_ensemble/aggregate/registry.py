"""Registry for aggregation strategies.

Central registry that maps strategy names to adapter classes.
Strategies self-register via decorator at import time.
"""

from __future__ import annotations
from typing import Dict, Type, Callable

from llm_ensemble.aggregate.ports.aggregation_strategy import AggregationStrategyPort


class AggregationStrategyRegistry:
    """Registry for aggregation strategy adapters."""
    
    _strategies: Dict[str, Type[AggregationStrategyPort]] = {}
    
    @classmethod
    def register(cls, name: str) -> Callable:
        """Decorator to register a strategy adapter.
        
        Args:
            name: Strategy name (natural key, e.g., 'majority_vote')
            
        Returns:
            Decorator function
            
        Example:
            @AggregationStrategyRegistry.register("majority_vote")
            class MajorityVoteAdapter(AggregationStrategyPort):
                ...
        """
        def decorator(adapter_class: Type[AggregationStrategyPort]) -> Type[AggregationStrategyPort]:
            if name in cls._strategies:
                raise ValueError(f"Aggregation strategy '{name}' already registered")
            cls._strategies[name] = adapter_class
            return adapter_class
        return decorator
    
    @classmethod
    def get(cls, name: str) -> AggregationStrategyPort:
        """Get strategy adapter instance by name.
        
        Args:
            name: Strategy name
            
        Returns:
            Instantiated strategy adapter
            
        Raises:
            ValueError: If strategy not found in registry
        """
        if name not in cls._strategies:
            available = ", ".join(sorted(cls._strategies.keys()))
            raise ValueError(
                f"Aggregation strategy '{name}' not found in registry. "
                f"Available: {available}"
            )
        adapter_class = cls._strategies[name]
        return adapter_class(strategy_name=name)
    
    @classmethod
    def list_strategies(cls) -> list[str]:
        """List all registered strategy names.
        
        Returns:
            Sorted list of strategy names
        """
        return sorted(cls._strategies.keys())
    
    @classmethod
    def has_strategy(cls, name: str) -> bool:
        """Check if strategy is registered.
        
        Args:
            name: Strategy name
            
        Returns:
            True if strategy exists in registry
        """
        return name in cls._strategies
