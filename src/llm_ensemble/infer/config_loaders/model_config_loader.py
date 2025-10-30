"""Model configuration loader.

Loads YAML model configuration files and returns ModelConfig domain objects.
"""

from __future__ import annotations
import os

from llm_ensemble.infer.schemas.model_config_schema import ModelConfig
from llm_ensemble.libs.config import load_yaml_config
from llm_ensemble.libs.runtime.path_manager import PathManager


def load_model_config(model_id: str) -> ModelConfig:
    """Load a model configuration from YAML file.

    Args:
        model_id: Model identifier (e.g., "phi3-mini", "gpt-oss-20b")

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
    return load_yaml_config(
        config_name=model_id,
        config_dir=PathManager.get_model_configs_dir(),
        schema=ModelConfig,
        config_type="model",
    )


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
