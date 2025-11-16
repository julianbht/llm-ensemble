"""Config override utilities.

Provides utilities for parsing and applying CLI overrides to configuration objects.
This enables users to override specific config values without creating new config files,
while maintaining full reproducibility by tracking all overrides in manifests.

Supports prefix-based routing for multi-config CLIs:
    --override model.temperature=0.7
    --override prompt.variables.role=false
    --override io.reader=custom_reader
"""

from __future__ import annotations
from typing import Any, Literal
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
    """
    for key, value in updates.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def parse_and_route_overrides(
    override_list: list[str],
    valid_prefixes: list[str] | None = None
) -> dict[str, dict]:
    """Parse overrides with config prefixes and route to appropriate configs.

    Uses prefix-based routing to explicitly target different configuration types.
    This eliminates ambiguity and hardcoded field lists.

    Args:
        override_list: List of "prefix.key=value" strings from CLI
        valid_prefixes: Optional list of valid prefixes (defaults to ['model', 'prompt', 'io'])

    Returns:
        Dict mapping config types to their overrides
        Example: {'model': {'temperature': 0.7}, 'prompt': {'variables': {'role': False}}}

    Raises:
        ValueError: If override format is invalid or uses unknown prefix
    """
    if valid_prefixes is None:
        valid_prefixes = ['model', 'prompt', 'io']

    # Initialize result dict with all valid prefixes
    routed = {prefix: {} for prefix in valid_prefixes}

    for override in override_list:
        if "=" not in override:
            raise ValueError(
                f"Invalid override format: '{override}'. Expected 'config.key=value'"
            )

        key_path, value = override.split("=", 1)
        parts = key_path.split(".", 1)

        if len(parts) < 2:
            raise ValueError(
                f"Override must specify config prefix: '{override}'\n"
                f"Expected format: '<config>.key=value'\n"
                f"Valid prefixes: {', '.join(valid_prefixes)}\n"
                f"Examples:\n"
                f"  --override model.temperature=0.7\n"
                f"  --override prompt.variables.role=false\n"
                f"  --override io.reader=custom_reader"
            )

        config_type, rest = parts

        if config_type not in routed:
            raise ValueError(
                f"Unknown config type: '{config_type}'\n"
                f"Valid prefixes: {', '.join(valid_prefixes)}\n"
                f"Did you mean one of: {', '.join(valid_prefixes)}?"
            )

        # Parse the rest as nested overrides (e.g., "variables.role" → {"variables": {"role": ...}})
        nested_override = parse_overrides([f"{rest}={value}"])

        # Merge into the config bucket
        _deep_update(routed[config_type], nested_override)

    return routed
