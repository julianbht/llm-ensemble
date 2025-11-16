"""Validation callbacks for CLI parameters with rich error messages.

This module provides Typer callback functions that validate required parameters
and show helpful error messages listing available options when validation fails.
"""

from pathlib import Path
import typer

from llm_ensemble.libs.runtime.path_manager import PathManager
from llm_ensemble.libs.cli.error_messages import (
    format_missing_config_error,
    format_missing_io_config_error,
    format_missing_ensemble_config_error,
)


def list_available_configs(config_dir: Path) -> list[str]:
    """List available YAML configs in directory.
    
    Args:
        config_dir: Directory to scan for .yaml files
    
    Returns:
        Sorted list of config names (without .yaml extension)
    """
    if not config_dir.exists():
        return []
    return sorted([p.stem for p in config_dir.glob("*.yaml")])


def validate_model_config(value: str) -> str:
    """Validate model config and show available options if missing.
    
    Args:
        value: Model config name
    
    Returns:
        The validated config name
    
    Raises:
        typer.BadParameter: If value is empty, with list of available configs
    """
    if not value:
        config_dir = PathManager.get_model_configs_dir()
        available = list_available_configs(config_dir)
        example = available[0] if available else "gpt-oss-20b-free"
        raise typer.BadParameter(
            format_missing_config_error(
                param_name="--model-cfg",
                config_type="model",
                config_dir=config_dir,
                available=available,
                example=example,
            )
        )
    return value


def validate_prompt_config(value: str) -> str:
    """Validate prompt config and show available options if missing.
    
    Args:
        value: Prompt config name
    
    Returns:
        The validated config name
    
    Raises:
        typer.BadParameter: If value is empty, with list of available configs
    """
    if not value:
        config_dir = PathManager.get_prompts_dir()
        available = list_available_configs(config_dir)
        example = available[0] if available else "thomas-simple"
        raise typer.BadParameter(
            format_missing_config_error(
                param_name="--prompt-cfg",
                config_type="prompt",
                config_dir=config_dir,
                available=available,
                example=example,
            )
        )
    return value


def validate_ensemble_config(value: str) -> str:
    """Validate ensemble config and show available options if missing.
    
    Args:
        value: Ensemble config name
    
    Returns:
        The validated config name
    
    Raises:
        typer.BadParameter: If value is empty, with list of available configs
    """
    if not value:
        config_dir = PathManager.get_ensembles_dir()
        available = list_available_configs(config_dir)
        raise typer.BadParameter(
            format_missing_ensemble_config_error(config_dir, available)
        )
    return value


def validate_ingest_io_config(value: str) -> str:
    """Validate I/O config for ingest CLI and show available options if missing.
    
    Args:
        value: I/O config name
    
    Returns:
        The validated config name
    
    Raises:
        typer.BadParameter: If value is empty, with list of available configs
    """
    if not value:
        config_dir = PathManager.get_io_configs_dir("ingest")
        available = list_available_configs(config_dir)
        raise typer.BadParameter(
            format_missing_io_config_error("ingest", config_dir, available)
        )
    return value


def validate_infer_io_config(value: str) -> str:
    """Validate I/O config for infer CLI and show available options if missing.
    
    Args:
        value: I/O config name
    
    Returns:
        The validated config name
    
    Raises:
        typer.BadParameter: If value is empty, with list of available configs
    """
    if not value:
        config_dir = PathManager.get_io_configs_dir("infer")
        available = list_available_configs(config_dir)
        raise typer.BadParameter(
            format_missing_io_config_error("infer", config_dir, available)
        )
    return value


def validate_aggregate_io_config(value: str) -> str:
    """Validate I/O config for aggregate CLI and show available options if missing.
    
    Args:
        value: I/O config name
    
    Returns:
        The validated config name
    
    Raises:
        typer.BadParameter: If value is empty, with list of available configs
    """
    if not value:
        config_dir = PathManager.get_io_configs_dir("aggregate")
        available = list_available_configs(config_dir)
        raise typer.BadParameter(
            format_missing_io_config_error("aggregate", config_dir, available)
        )
    return value


def validate_evaluate_io_config(value: str) -> str:
    """Validate I/O config for evaluate CLI and show available options if missing.
    
    Args:
        value: I/O config name
    
    Returns:
        The validated config name
    
    Raises:
        typer.BadParameter: If value is empty, with list of available configs
    """
    if not value:
        config_dir = PathManager.get_io_configs_dir("evaluate")
        available = list_available_configs(config_dir)
        raise typer.BadParameter(
            format_missing_io_config_error("evaluate", config_dir, available)
        )
    return value
