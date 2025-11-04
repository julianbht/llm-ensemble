"""Ensemble configuration schema."""

from __future__ import annotations
from typing import Optional, Any
from pydantic import Field

from llm_ensemble.libs.schemas.base_config import BaseConfig


class EnsembleConfig(BaseConfig):
    """Configuration for ensemble aggregation strategies.
    
    Specifies which strategy to use via dynamic adapter loading.
    
    Example YAML:
        strategy: majority_vote
        strategy_module: llm_ensemble.aggregate.adapters.strategies.majority_vote_adapter
        strategy_class: MajorityVoteAdapter
    
    Future enhancement: Add tie_breaking_strategy parameter for configurable tie resolution.
    """
    
    strategy: str = Field(
        ...,
        description=(
            "Name of the aggregation strategy to use. "
            "Supported: 'majority_vote', 'weighted_majority'"
        )
    )
    
    # Dynamic adapter loading
    strategy_module: str = Field(
        ...,
        description="Full Python module path to strategy adapter (e.g., 'llm_ensemble.aggregate.adapters.strategies.majority_vote_adapter')"
    )
    strategy_class: str = Field(
        ...,
        description="Strategy adapter class name in UpperCamelCase (e.g., 'MajorityVoteAdapter')"
    )
    
    # Optional name hint for run_name generation (derived from filename by loader)
    name_hint: Optional[str] = Field(
        default=None,
        description="Short name hint for run_name generation (e.g., 'majority_vote')"
    )
    
    def get_strategy(self) -> Any:
        """Instantiate and return the aggregation strategy adapter.
        
        Dynamically imports the strategy module and instantiates the strategy class.
        
        Returns:
            Instance of the aggregation strategy adapter
            
        Raises:
            ImportError: If the strategy module cannot be imported
            AttributeError: If the strategy class doesn't exist in the module
            
        Example:
            >>> config = EnsembleConfig(
            ...     strategy="majority_vote",
            ...     strategy_module="llm_ensemble.aggregate.adapters.strategies.majority_vote_adapter",
            ...     strategy_class="MajorityVoteAdapter"
            ... )
            >>> strategy = config.get_strategy()
        """
        return self._instantiate_adapter(self.strategy_module, self.strategy_class)
