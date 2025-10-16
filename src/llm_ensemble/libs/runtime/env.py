"""Runtime environment configuration loader.

Loads environment-specific configuration from configs/runtime/*.env files
based on the APP_ENV environment variable.

This follows 12-factor app principles by separating environment-specific
configuration from code.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional


def load_runtime_config(app_env: Optional[str] = None) -> None:
    """Load environment-specific configuration from configs/runtime/*.env.

    Loads the appropriate .env file based on APP_ENV and sets environment
    variables. This should be called early in CLI initialization.

    Args:
        app_env: Environment name (dev, prod, ci). Defaults to APP_ENV env var or "dev"

    Example:
        >>> load_runtime_config()  # Loads configs/runtime/dev.env by default
        >>> load_runtime_config("prod")  # Loads configs/runtime/prod.env

    Environment precedence (highest to lowest):
        1. Already-set environment variables (never overwritten)
        2. Runtime config file (configs/runtime/{env}.env)
        3. Default values in code

    Note:
        This function does NOT overwrite existing environment variables.
        This allows manual overrides via shell environment or .env files.
    """
    # Determine which environment to load
    if app_env is None:
        app_env = os.getenv("APP_ENV", "dev")

    # Find configs/runtime directory (relative to project root)
    # This file is at src/llm_ensemble/libs/runtime/env.py
    # Project root is 4 levels up
    project_root = Path(__file__).parents[4]
    runtime_config_dir = project_root / "configs" / "runtime"
    config_file = runtime_config_dir / f"{app_env}.env"

    if not config_file.exists():
        # Silent fallback - don't error if config file missing
        # This allows the system to work with just env vars
        return

    # Parse and load environment variables
    with open(config_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Parse KEY=VALUE
            if "=" not in line:
                continue

            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()

            # Remove inline comments
            if "#" in value:
                value = value.split("#")[0].strip()

            # Only set if not already in environment (precedence rule)
            if key and key not in os.environ:
                os.environ[key] = value


def get_app_env() -> str:
    """Get the current application environment.

    Returns:
        Current environment name (dev, prod, ci, etc.)

    Example:
        >>> get_app_env()
        'dev'
    """
    return os.getenv("APP_ENV", "dev")


def is_production() -> bool:
    """Check if running in production environment.

    Returns:
        True if APP_ENV is "prod", False otherwise

    Example:
        >>> is_production()
        False
    """
    return get_app_env() == "prod"


def is_ci() -> bool:
    """Check if running in CI/CD environment.

    Returns:
        True if APP_ENV is "ci", False otherwise

    Example:
        >>> is_ci()
        False
    """
    return get_app_env() == "ci"
