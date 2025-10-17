"""Model configuration loader.

Loads YAML model configuration files and returns ModelConfig domain objects.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
import yaml

from llm_ensemble.infer.config.models import ModelConfig


def get_default_config_dir() -> Path:
    """Get the default configs/models directory.

    Returns:
        Path to configs/models relative to project root
    """
    # Navigate from this file to project root, then to configs/models
    # This file is at: src/llm_ensemble/infer/config/loader.py
    # Project root is 4 levels up
    project_root = Path(__file__).parents[4]
    return project_root / "configs" / "models"


def load_model_config(
    model_id: str,
    config_dir: Optional[Path] = None,
) -> ModelConfig:
    """Load a model configuration from YAML file.

    Args:
        model_id: Model identifier (e.g., "phi3-mini", "gpt-oss-20b")
        config_dir: Directory containing model configs (defaults to configs/models)

    Returns:
        ModelConfig object with all settings loaded from YAML

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If YAML is invalid or missing required fields

    Example:
        >>> config = load_model_config("gpt-oss-20b")
        >>> config.provider
        'openrouter'
        >>> config.openrouter_model_id
        'openai/gpt-oss-20b:free'
    """
    # Determine config directory
    if config_dir is None:
        config_dir = get_default_config_dir()

    # Build path to config file
    config_path = config_dir / f"{model_id}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Model config not found: {config_path}\n"
            f"Available configs in {config_dir}:\n"
            + "\n".join(f"  - {p.stem}" for p in config_dir.glob("*.yaml"))
        )

    # Load YAML
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid config file {config_path}: expected YAML object")

    # Validate and parse into ModelConfig
    try:
        return ModelConfig(**data)
    except Exception as e:
        raise ValueError(f"Failed to parse config {config_path}: {e}") from e


def get_endpoint_url(model_config: ModelConfig) -> str:
    """Get the API endpoint URL for a model.

    Determines the endpoint based on provider and configuration.
    Supports environment variable overrides.

    Args:
        model_config: ModelConfig object

    Returns:
        Endpoint URL string

    Raises:
        ValueError: If endpoint cannot be determined

    Example:
        >>> config = load_model_config("gpt-oss-20b")
        >>> get_endpoint_url(config)
        'https://openrouter.ai/api/v1'
    """
    if model_config.provider == "openrouter":
        return "https://openrouter.ai/api/v1"

    elif model_config.provider == "hf":
        # Check for environment variable override
        env_var = f"HF_ENDPOINT_{model_config.model_id.upper().replace('-', '_')}_URL"
        if env_var in os.environ:
            return os.environ[env_var]

        # Use explicit endpoint URL if provided
        if model_config.hf_endpoint_url:
            return model_config.hf_endpoint_url

        # Fall back to public inference API
        if model_config.hf_model_name:
            return f"https://api-inference.huggingface.co/models/{model_config.hf_model_name}"

        raise ValueError(
            f"HuggingFace model {model_config.model_id} requires either "
            f"hf_endpoint_url or hf_model_name in config"
        )

    elif model_config.provider == "ollama":
        # Ollama typically runs locally
        return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    else:
        raise ValueError(f"Unsupported provider: {model_config.provider}")
