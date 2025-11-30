"""Aggregation strategy configuration schema.

Complete configuration for aggregation strategies.
All configuration centralized here - adapters contain no metadata.
"""

from __future__ import annotations
from typing import Any
from pydantic import Field, BaseModel

from llm_ensemble.libs.schemas.base_config import BaseConfig


class AggregationStrategyAdapterConfig(BaseModel):
    """Nested config for adapter instantiation details."""

    aggregation_strategy_module: str = Field(
        ...,
        description="Full Python module path to strategy adapter"
    )
    aggregation_strategy_class: str = Field(
        ...,
        description="Strategy adapter class name in UpperCamelCase"
    )


class AggregationStrategyConfig(BaseConfig):
    """Complete configuration for aggregation strategy.

    All strategy config centralized here - adapters are pure implementation.
    This config includes both the strategy identity AND adapter wiring.

    Example YAML:
        name_hint: majority-vote
        aggregation_strategy_name: majority_vote
        aggregation_strategy_adapter:
            aggregation_strategy_module: llm_ensemble.aggregate.adapters.strategies.majority_vote_adapter
            aggregation_strategy_class: MajorityVoteAdapter

    Note: name_hint is inherited from BaseConfig and used for run_name generation.
    """

    aggregation_strategy_name: str = Field(
        ...,
        description="Natural key for AggregationStrategy entity (e.g., 'majority_vote')"
    )

    aggregation_strategy_adapter: AggregationStrategyAdapterConfig = Field(
        ...,
        description="Adapter instantiation configuration"
    )

    def get_adapter(self) -> Any:
        """Instantiate and return the aggregation strategy adapter.

        Dynamically imports the strategy module and instantiates the strategy class.
        Strategy name comes from config and is passed to adapter constructor.

        Returns:
            Instance of the aggregation strategy adapter

        Raises:
            ImportError: If the strategy module cannot be imported
            AttributeError: If the strategy class doesn't exist in the module
        """
        return self._instantiate_adapter(
            self.aggregation_strategy_adapter.aggregation_strategy_module,
            self.aggregation_strategy_adapter.aggregation_strategy_class,
            strategy_name=self.aggregation_strategy_name
        )
