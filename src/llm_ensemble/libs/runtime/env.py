"""Runtime environment configuration loader.

Loads configuration using python-dotenv with a two-layer approach:
1. Root .env (gitignored) - for secrets and local overrides
2. configs/runtime/*.env (committed) - for non-secret defaults

This follows 12-factor app principles by separating environment-specific
configuration from code while keeping secrets out of version control.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv


def load_runtime_config(app_env: Optional[str] = None) -> None:
    """Load environment configuration using python-dotenv.

    Loads configuration in two layers:
    1. Root .env file (gitignored) - secrets and local overrides
    2. configs/runtime/{env}.env - environment-specific non-secret defaults

    Args:
        app_env: Environment name (dev, prod, ci). Defaults to APP_ENV env var or "dev"

    Example:
        >>> load_runtime_config()  # Loads .env + configs/runtime/dev.env
        >>> load_runtime_config("prod")  # Loads .env + configs/runtime/prod.env

    Environment precedence (highest to lowest):
        1. Already-set shell environment variables
        2. Root .env file (secrets, local overrides)
        3. Runtime config file configs/runtime/{env}.env (non-secret defaults)

    Note:
        python-dotenv does NOT overwrite existing environment variables.
        This allows manual overrides via shell environment.
    """
    # Find project root (4 levels up from this file)
    project_root = Path(__file__).parents[4]

    # Layer 1: Load root .env first (secrets and local overrides)
    # override=False means existing env vars take precedence
    root_env = project_root / ".env"
    if root_env.exists():
        load_dotenv(dotenv_path=root_env, override=False)

    # Determine which environment to load (may have been set in root .env)
    if app_env is None:
        app_env = os.getenv("APP_ENV", "dev")

    # Layer 2: Load environment-specific config (non-secret defaults)
    runtime_config_file = project_root / "configs" / "runtime" / f"{app_env}.env"
    if runtime_config_file.exists():
        load_dotenv(dotenv_path=runtime_config_file, override=False)


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
