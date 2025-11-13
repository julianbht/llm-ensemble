"""Generic YAML configuration loader with Pydantic validation.

Provides a single, reusable implementation for loading YAML configuration files
and validating them against Pydantic schemas. Used by all CLIs to eliminate
code duplication across config loaders.
"""

from __future__ import annotations
from pathlib import Path
from typing import Type, TypeVar
import yaml

from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


def load_yaml_config(
    config_name: str,
    config_dir: Path,
    schema: Type[T],
    config_type: str = "configuration",
) -> T:
    """Load and validate a YAML configuration file.

    Generic loader that:
    1. Resolves config file path from name + directory
    2. Loads YAML content
    3. Validates against Pydantic schema
    4. Provides helpful error messages with available configs

    Args:
        config_name: Configuration identifier (e.g., "gpt-oss-20b", "ndjson")
        config_dir: Directory containing config files
        schema: Pydantic model class to parse into
        config_type: Human-readable type name for error messages (e.g., "model", "prompt")

    Returns:
        Validated Pydantic model instance

    Raises:
        FileNotFoundError: If config file doesn't exist (lists available configs)
        ValueError: If YAML is invalid or fails schema validation

    Example:
        >>> from llm_ensemble.infer.schemas import ModelConfig
        >>> from llm_ensemble.libs.runtime.path_manager import PathManager
        >>> config = load_yaml_config(
        ...     "gpt-oss-20b",
        ...     PathManager.get_model_configs_dir(),
        ...     ModelConfig,
        ...     "model"
        ... )
        >>> config.provider
        'openrouter'
    """
    # Build path to config file
    config_path = config_dir / f"{config_name}.yaml"

    # Check if file exists, provide helpful error with available configs
    if not config_path.exists():
        available = [p.stem for p in config_dir.glob("*.yaml")] if config_dir.exists() else []
        available_list = "\n".join(f"  - {fmt}" for fmt in sorted(available))
        raise FileNotFoundError(
            f"{config_type.capitalize()} config not found: {config_path}\n"
            f"Available {config_type} configs in {config_dir}:\n"
            f"{available_list}"
        )

    # Load YAML
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Validate structure
    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid config file {config_path}: expected YAML object, got {type(data).__name__}"
        )

    # Inject 'name' field from filename (configs inherit from BaseConfig which has this field)
    # Only inject if not already present in YAML (allow explicit override for testing)
    if 'name' not in data or data['name'] is None:
        data['name'] = config_name

    # Parse and validate against Pydantic schema
    try:
        return schema(**data)
    except Exception as e:
        raise ValueError(
            f"Failed to parse {config_type} config {config_path}: {e}"
        ) from e
