"""Configuration loader for inference pipeline.

Startup Layer - Configuration Management

Responsible for loading configuration from YAML files and constructing
domain entities. This keeps the domain models pure (no load methods on schemas).

Logging configuration is read from environment variables during application execution.
"""

from __future__ import annotations

from llm_ensemble.infer.domain.entities.model_config import ModelConfig
from llm_ensemble.infer.domain.entities.retry_config_schema import RetryConfig
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


def build_run_name_hints(
    model_config_name: str,
    prompt_template_name: str,
    provider_name: str,
    io_name: str,
) -> list[str]:
    """Build run name hints from configuration names.

    Application logic for determining what makes a meaningful run name.
    Loads minimal config data needed for name hint extraction.

    Args:
        model_config_name: Model config identifier
        prompt_template_name: Prompt template name
        provider_name: Provider name
        io_name: I/O adapter name

    Returns:
        List of name hints for run name generation
    """
    # Load model config to extract name hint
    model_cfg = load_model_config(model_config_name)

    return [
        model_cfg.name_hint,
        prompt_template_name,
        provider_name,
        io_name,
    ]
