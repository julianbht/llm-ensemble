"""Logging configuration factory.

Factory for loading and validating logging configuration from YAML files in configs/logging/.
"""

from __future__ import annotations

from llm_ensemble.libs.schemas.logging_config import LoggingConfig
from llm_ensemble.libs.config import load_yaml_config
from llm_ensemble.libs.runtime.path_manager import PathManager


class LoggingConfigFactory:
    """Factory for loading logging configurations from YAML files."""

    @staticmethod
    def load(config_name: str) -> LoggingConfig:
        """Load and validate logging configuration from YAML file.

        Args:
            config_name: Name of the logging config file (without .yaml extension)
                        e.g., "standard", "json", "console-only"

        Returns:
            Validated LoggingConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config validation fails
        """
        return load_yaml_config(
            config_name=config_name,
            config_dir=PathManager.get_configs_dir() / "logging",
            schema=LoggingConfig,
            config_type="logging",
        )
