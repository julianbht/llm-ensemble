"""Ensemble configuration schema."""

from __future__ import annotations
from typing import Any
from uuid import UUID
from pydantic import Field

from llm_ensemble.libs.schemas.base_config import BaseConfig
from llm_ensemble.libs.db import compute_aggregation_spec_uuid


class EnsembleConfig(BaseConfig):
    """Configuration entity for ensemble aggregation strategies.

    Specifies which strategy to use via dynamic adapter loading.
    Config is persisted as AggregationSpecORM with deterministic UUID.

    Example YAML:
        strategy_module: llm_ensemble.aggregate.adapters.strategies.majority_vote_adapter
        strategy_class: MajorityVoteAdapter

    Note: name_hint is inherited from BaseConfig and typically derived from filename.

    Future enhancement: Add tie_breaking_strategy parameter for configurable tie resolution.
    """

    id: UUID = Field(
        ...,
        description="Deterministic UUID computed from config name (natural key)"
    )

    # Dynamic adapter loading
    strategy_module: str = Field(
        ...,
        description="Full Python module path to strategy adapter"
    )
    strategy_class: str = Field(
        ...,
        description="Strategy adapter class name in UpperCamelCase"
    )

    @classmethod
    def create(cls, name: str, **kwargs) -> "EnsembleConfig":
        """Create EnsembleConfig with computed ID from name.

        Args:
            name: Config name (natural key, typically from filename)
            **kwargs: Other config fields (strategy_module, strategy_class, etc.)

        Returns:
            EnsembleConfig with computed ID
        """
        config_id = compute_aggregation_spec_uuid(name)
        return cls(id=config_id, name=name, **kwargs)

    def get_strategy(self) -> Any:
        """Instantiate and return the aggregation strategy adapter.

        Dynamically imports the strategy module and instantiates the strategy class
        with the aggregation_spec_id (from this config's ID).

        Returns:
            Instance of the aggregation strategy adapter

        Raises:
            ImportError: If the strategy module cannot be imported
            AttributeError: If the strategy class doesn't exist in the module
        """
        return self._instantiate_adapter(
            self.strategy_module,
            self.strategy_class,
            aggregation_spec_id=self.id
        )
