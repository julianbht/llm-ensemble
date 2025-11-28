"""Configuration loaders for aggregate CLI."""

from __future__ import annotations

from llm_ensemble.aggregate.schemas.ensemble_config_schema import EnsembleConfig
from llm_ensemble.libs.runtime.path_manager import PathManager
from llm_ensemble.libs.config.yaml_config_loader import load_yaml_config


def load_ensemble_config(config_name: str) -> EnsembleConfig:
    """Load ensemble configuration from YAML file.

    Args:
        config_name: Name of the ensemble config (without .yaml extension)

    Returns:
        Parsed and validated EnsembleConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    config_dir = PathManager.get_project_root() / "configs" / "ensembles"
    return load_yaml_config(
        config_name=config_name,
        config_dir=config_dir,
        schema=EnsembleConfig,
        config_type="ensemble",
    )
