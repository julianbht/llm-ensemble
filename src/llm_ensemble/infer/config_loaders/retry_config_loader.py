"""Retry configuration loader.

Loads YAML retry configuration files and returns RetryConfig domain objects.
"""

from __future__ import annotations

from llm_ensemble.infer.schemas.retry_config_schema import RetryConfig
from llm_ensemble.libs.config import load_yaml_config
from llm_ensemble.libs.runtime.path_manager import PathManager


def load_retry_config(retry_id: str) -> RetryConfig:
    """Load a retry configuration from YAML file.

    Args:
        retry_id: Retry configuration identifier (e.g., "standard", "aggressive")

    Returns:
        RetryConfig object with all settings loaded from YAML

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If YAML is invalid or missing required fields

    Example:
        >>> config = load_retry_config("standard")
        >>> config.max_retries
        5
        >>> config.base_delay_seconds
        1.0
    """
    return load_yaml_config(
        config_name=retry_id,
        config_dir=PathManager.get_retries_dir(),
        schema=RetryConfig,
        config_type="retry",
    )
