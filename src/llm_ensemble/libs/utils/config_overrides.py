"""Config override utilities.

Provides utilities for parsing and applying CLI overrides to configuration objects.
This enables users to override specific config values without creating new config files,
while maintaining full reproducibility by tracking all overrides in manifests.
"""

from __future__ import annotations
from typing import Any
from pydantic import BaseModel


def parse_overrides(override_list: list[str]) -> dict[str, Any]:
    """Parse --override flags into nested dict.

    Supports nested keys using dot notation (e.g., "additional_params.top_k=50").
    Automatically converts values to appropriate types (bool, int, float, None, str).

    Args:
        override_list: List of "key=value" strings from CLI

    Returns:
        Nested dict of overrides

    Raises:
        ValueError: If override format is invalid

    Examples:
        >>> parse_overrides(["temperature=0.7", "max_tokens=512"])
        {'temperature': 0.7, 'max_tokens': 512}

        >>> parse_overrides(["additional_params.top_k=50"])
        {'additional_params': {'top_k': 50}}

        >>> parse_overrides(["variables.role=false", "variables.aspects=true"])
        {'variables': {'role': False, 'aspects': True}}
    """
    result = {}

    for override in override_list:
        if "=" not in override:
            raise ValueError(
                f"Invalid override format: '{override}'. "
                f"Expected 'key=value' (e.g., 'temperature=0.7')"
            )

        key, value = override.split("=", 1)

        # Handle nested keys (e.g., "default_params.temperature")
        keys = key.split(".")

        # Parse value to appropriate type
        parsed_value = _parse_value(value)

        # Build nested dict
        current = result
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = parsed_value

    return result


def apply_overrides(config: BaseModel, overrides: dict[str, Any]) -> BaseModel:
    """Apply overrides to a Pydantic config model.

    Creates a new config instance with overrides applied, maintaining type validation.

    Args:
        config: Original config model (e.g., ModelConfig, PromptConfig)
        overrides: Dict of override values from parse_overrides()

    Returns:
        New config instance with overrides applied and validated

    Raises:
        ValidationError: If overrides result in invalid config

    Example:
        >>> model_config = load_model_config("gpt-oss-20b")
        >>> overrides = parse_overrides(["temperature=0.7", "seed=123"])
        >>> new_config = apply_overrides(model_config, overrides)
        >>> new_config.temperature
        0.7
    """
    # Convert to dict
    config_dict = config.model_dump()

    # Deep merge overrides
    _deep_update(config_dict, overrides)

    # Re-validate with Pydantic (ensures types are correct)
    return config.__class__(**config_dict)


def _parse_value(value: str) -> Any:
    """Parse string value to appropriate Python type.

    Args:
        value: String value from CLI

    Returns:
        Parsed value as bool, None, int, float, or str

    Examples:
        >>> _parse_value("true")
        True
        >>> _parse_value("false")
        False
        >>> _parse_value("null")
        None
        >>> _parse_value("42")
        42
        >>> _parse_value("3.14")
        3.14
        >>> _parse_value("hello")
        'hello'
    """
    # Boolean
    if value.lower() in ("true", "false"):
        return value.lower() == "true"

    # None/null
    if value.lower() in ("none", "null"):
        return None

    # Number (int or float)
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        pass

    # String (default)
    return value


def _deep_update(base: dict, updates: dict) -> None:
    """Deep merge updates into base dict (in-place).

    Args:
        base: Base dictionary to update
        updates: Updates to merge in

    Example:
        >>> base = {"a": 1, "b": {"c": 2}}
        >>> _deep_update(base, {"b": {"d": 3}})
        >>> base
        {'a': 1, 'b': {'c': 2, 'd': 3}}
    """
    for key, value in updates.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
