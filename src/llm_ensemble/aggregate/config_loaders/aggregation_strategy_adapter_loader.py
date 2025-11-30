"""Aggregation strategy config loader for aggregate CLI."""

from __future__ import annotations

from llm_ensemble.aggregate.schemas.aggregation_strategy_adapter_spec import AggregationStrategyConfig
from llm_ensemble.libs.runtime.path_manager import PathManager
from llm_ensemble.libs.config.yaml_config_loader import load_yaml_config


def load_aggregation_strategy_config(config_name: str) -> AggregationStrategyConfig:
    """Load aggregation strategy configuration from YAML file.

    Args:
        config_name: Name of the strategy config (without .yaml extension)

    Returns:
        Parsed and validated AggregationStrategyConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    config_dir = PathManager.get_project_root() / "configs" / "strategies"
    return load_yaml_config(
        config_name=config_name,
        config_dir=config_dir,
        schema=AggregationStrategyConfig,
        config_type="aggregation_strategy_config",
    )
