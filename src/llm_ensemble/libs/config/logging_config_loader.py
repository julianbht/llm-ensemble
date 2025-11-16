"""Logging configuration loader.

Loads and validates logging configuration from YAML files in configs/logging/.
"""

from __future__ import annotations
from pathlib import Path
import yaml
from llm_ensemble.libs.schemas.logging_config import LoggingConfig
from llm_ensemble.libs.runtime.path_manager import PathManager


def load_logging_config(config_name: str) -> LoggingConfig:
    """Load and validate logging configuration from YAML file.

    Args:
        config_name: Name of the logging config file (without .yaml extension)
                    e.g., "default", "json", "console-only"

    Returns:
        Validated LoggingConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config validation fails
    """
    # Get logging configs directory
    configs_dir = PathManager.get_configs_dir() / "logging"
    config_path = configs_dir / f"{config_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Logging config not found: {config_path}\n"
            f"Available configs in {configs_dir}: "
            f"{', '.join(p.stem for p in configs_dir.glob('*.yaml'))}"
        )

    # Load YAML
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    # Validate with Pydantic
    try:
        return LoggingConfig(**config_dict)
    except Exception as e:
        raise ValueError(
            f"Invalid logging config in {config_path}: {e}"
        ) from e
