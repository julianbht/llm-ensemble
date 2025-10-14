"""Configuration loader for model configs.

Reads YAML files from configs/models/ and loads them into ModelConfig objects.
Follows 12-factor app design: configuration via files, overridable by env vars.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

from llm_ensemble.infer.domain.models import ModelConfig


def load_model_config(model_id: str, config_dir: Optional[Path] = None) -> ModelConfig:
    """Load model configuration from YAML file.

    Args:
        model_id: Model identifier (e.g., 'phi3-mini')
        config_dir: Optional config directory path (defaults to configs/models/)

    Returns:
        ModelConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If YAML is invalid

    Example:
        >>> config = load_model_config("phi3-mini")
        >>> print(config.provider)
        'hf'
    """
    if config_dir is None:
        # Default to configs/models/ relative to project root
        # Assume we're in src/llm_ensemble/infer/adapters/config_loader.py
        # Navigate up to project root: ../../../../configs/models/
        this_file = Path(__file__).resolve()
        project_root = this_file.parent.parent.parent.parent.parent
        config_dir = project_root / "configs" / "models"

    config_path = config_dir / f"{model_id}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Model config not found: {config_path}. "
            f"Available configs: {list(config_dir.glob('*.yaml'))}"
        )

    try:
        import yaml
    except ImportError:
        raise RuntimeError(
            "PyYAML required for config loading. Install with: pip install pyyaml"
        )

    with config_path.open("r") as f:
        data = yaml.safe_load(f)

    # Environment variable overrides for HuggingFace endpoints
    # Follows 12-factor: config in env vars takes precedence
    if data.get("provider") == "hf":
        env_key_url = f"HF_ENDPOINT_{model_id.upper().replace('-', '_')}_URL"
        env_key_model = f"HF_ENDPOINT_{model_id.upper().replace('-', '_')}_MODEL"

        if env_key_url in os.environ:
            data["hf_endpoint_url"] = os.environ[env_key_url]

        if env_key_model in os.environ:
            data["hf_model_name"] = os.environ[env_key_model]

    return ModelConfig(**data)


def get_endpoint_url(model_config: ModelConfig) -> str:
    """Get the inference endpoint URL for a model.

    Args:
        model_config: Model configuration

    Returns:
        Endpoint URL

    Raises:
        ValueError: If endpoint URL is not configured

    Example:
        >>> config = load_model_config("phi3-mini")
        >>> url = get_endpoint_url(config)
    """
    if model_config.provider != "hf":
        raise ValueError(f"Only HuggingFace provider supported, got: {model_config.provider}")

    # Priority: explicit URL > model name > error
    if model_config.hf_endpoint_url:
        return model_config.hf_endpoint_url

    if model_config.hf_model_name:
        # Use HuggingFace Inference API
        base_url = "https://api-inference.huggingface.co/models"
        return f"{base_url}/{model_config.hf_model_name}"

    raise ValueError(
        f"No HuggingFace endpoint configured for {model_config.model_id}. "
        f"Set 'hf_endpoint_url' or 'hf_model_name' in config, or use environment variable "
        f"HF_ENDPOINT_{model_config.model_id.upper().replace('-', '_')}_URL"
    )
