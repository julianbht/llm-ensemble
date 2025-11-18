"""Ensemble configuration schema."""

from __future__ import annotations
from typing import Optional, Any
from pydantic import Field

from llm_ensemble.libs.schemas.base_config import BaseConfig


class EnsembleConfig(BaseConfig):
    """Configuration for ensemble aggregation strategies.

    Specifies which strategy to use via dynamic adapter loading.

    Example YAML:
        strategy_module: llm_ensemble.aggregate.adapters.strategies.majority_vote_adapter
        strategy_class: MajorityVoteAdapter

    Note: name_hint is inherited from BaseConfig and typically derived from filename.

    Future enhancement: Add tie_breaking_strategy parameter for configurable tie resolution.
    """

    # Dynamic adapter loading
    strategy_module: str = Field(
        ...,
        description="Full Python module path to strategy adapter"
    )
    strategy_class: str = Field(
        ...,
        description="Strategy adapter class name in UpperCamelCase"
    )

    def get_strategy(self) -> Any:
        """Instantiate and return the aggregation strategy adapter.
        
        Dynamically imports the strategy module and instantiates the strategy class.
        
        Returns:
            Instance of the aggregation strategy adapter
            
        Raises:
            ImportError: If the strategy module cannot be imported
            AttributeError: If the strategy class doesn't exist in the module
        """
        return self._instantiate_adapter(self.strategy_module, self.strategy_class)
