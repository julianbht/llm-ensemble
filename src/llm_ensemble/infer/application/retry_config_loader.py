"""Retry configuration factory.

Factory for loading YAML retry configuration files and returning RetryConfig domain objects.
"""

from __future__ import annotations

from llm_ensemble.infer.schemas.retry_config_schema import RetryConfig
from llm_ensemble.libs.config import load_yaml_config
from llm_ensemble.libs.runtime.path_manager import PathManager


class RetryConfigFactory:
    """Factory for loading retry configurations from YAML files."""

    @staticmethod
    def load(retry_id: str) -> RetryConfig:
        """Load a retry configuration from YAML file.

        Args:
            retry_id: Retry configuration identifier (e.g., "standard", "aggressive")

        Returns:
            RetryConfig object with all settings loaded from YAML

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If YAML is invalid or missing required fields
        """
        return load_yaml_config(
            config_name=retry_id,
            config_dir=PathManager.get_retries_dir(),
            schema=RetryConfig,
            config_type="retry",
        )
