"""Configuration loaders for aggregate CLI."""

from __future__ import annotations

from llm_ensemble.aggregate.schemas.strategy_adapter_config import StrategyAdapterConfig
from llm_ensemble.libs.runtime.path_manager import PathManager
from llm_ensemble.libs.config.yaml_config_loader import load_yaml_config


def load_strategy_adapter_config(config_name: str) -> StrategyAdapterConfig:
    """Load strategy adapter configuration from YAML file.

    Args:
        config_name: Name of the strategy adapter config (without .yaml extension)

    Returns:
        Parsed and validated StrategyAdapterConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    config_dir = PathManager.get_project_root() / "configs" / "strategies"
    return load_yaml_config(
        config_name=config_name,
        config_dir=config_dir,
        schema=StrategyAdapterConfig,
        config_type="strategy_adapter",
    )
