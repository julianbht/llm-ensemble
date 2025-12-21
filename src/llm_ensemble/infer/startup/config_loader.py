"""Configuration loader for inference pipeline.

Startup Layer - Configuration Management

Responsible for loading configuration from YAML files and constructing
domain entities. This keeps the domain models pure (no load methods on schemas).
"""

from __future__ import annotations
from pathlib import Path

from llm_ensemble.infer.domain.entities.model_config import ModelConfig
from llm_ensemble.infer.domain.entities.retry_config_schema import RetryConfig
from llm_ensemble.libs.schemas.logging_config import LoggingConfig
from llm_ensemble.libs.runtime.path_manager import PathManager
from llm_ensemble.libs.config.yaml_config_loader import load_yaml_config


def load_model_config(model_id: str) -> ModelConfig:
    """Load model configuration from YAML file.

    Args:
        model_id: Model identifier (e.g., "phi3-mini", "gpt-oss-20b")

    Returns:
        ModelConfig with all settings loaded from YAML

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If YAML is invalid
    """
    config_dir = PathManager.get_configs_dir() / "models"
    return load_yaml_config(
        config_name=model_id,
        config_dir=config_dir,
        schema=ModelConfig,
        config_type="ModelConfig",
    )


def load_retry_config(retry_id: str) -> RetryConfig:
    """Load retry configuration from YAML file.

    Args:
        retry_id: Retry configuration identifier (e.g., "standard", "aggressive")

    Returns:
        RetryConfig with all settings loaded from YAML

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If YAML is invalid
    """
    config_dir = PathManager.get_configs_dir() / "retry"
    return load_yaml_config(
        config_name=retry_id,
        config_dir=config_dir,
        schema=RetryConfig,
        config_type="RetryConfig",
    )


def load_logging_config(logging_id: str) -> LoggingConfig:
    """Load logging configuration from YAML file.

    Args:
        logging_id: Logging configuration identifier (e.g., "standard", "observability")

    Returns:
        LoggingConfig with all settings loaded from YAML

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If YAML is invalid
    """
    config_dir = PathManager.get_configs_dir() / "logging"
    return load_yaml_config(
        config_name=logging_id,
        config_dir=config_dir,
        schema=LoggingConfig,
        config_type="LoggingConfig",
    )
