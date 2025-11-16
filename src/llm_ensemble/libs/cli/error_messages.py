"""Rich error message formatters for CLI validation errors.

This module provides user-friendly error messages when required CLI parameters
are missing, including available options and usage examples.
"""

from pathlib import Path
from typing import List


def _relative_config_dir(config_dir: Path) -> Path:
    """Return config_dir relative to project root if possible."""
    from llm_ensemble.libs.runtime.path_manager import PathManager

    try:
        return config_dir.relative_to(PathManager.get_project_root())
    except ValueError:
        return config_dir


def _format_config_options_message(
    header_line: str,
    param_name: str,
    config_type: str,
    config_dir: Path,
    available: List[str],
    example: str,
) -> str:
    """Common formatter for config selection helper text."""
    relative_dir = _relative_config_dir(config_dir)

    lines = [
        f"\n{header_line}",
        f"\nAvailable {config_type} configs in {relative_dir}/:",
    ]

    if available:
        lines.extend([f"  • {name}" for name in available])
    else:
        lines.append(f"  (No {config_type} configs found)")

    lines.extend(
        [
            f"\nExample usage:",
            f"  <command> {param_name} {example} ...",
        ]
    )
    return "\n".join(lines)


def format_missing_config_error(
    param_name: str,
    config_type: str,
    config_dir: Path,
    available: List[str],
    example: str,
) -> str:
    """Format a rich error message for missing config parameter."""
    header = f"Missing required option: {param_name}"
    return _format_config_options_message(
        header, param_name, config_type, config_dir, available, example
    )


def format_invalid_config_error(
    param_name: str,
    config_type: str,
    config_dir: Path,
    available: List[str],
    example: str,
    invalid_value: str,
) -> str:
    """Format error message for invalid config selection."""
    header = f"Unknown {config_type} config '{invalid_value}'"
    return _format_config_options_message(
        header, param_name, config_type, config_dir, available, example
    )


def format_missing_io_config_error(
    cli_name: str,
    config_dir: Path,
    available: List[str],
) -> str:
    """Format error message for missing I/O config (CLI-specific).
    
    Args:
        cli_name: CLI name (e.g., "infer", "ingest")
        config_dir: Directory containing I/O configs
        available: List of available config names
    
    Returns:
        Formatted error message with available options
    """
    example = available[0] if available else "json"
    return format_missing_config_error(
        param_name="--io-cfg",
        config_type=f"{cli_name} I/O",
        config_dir=config_dir,
        available=available,
        example=example,
    )


def format_invalid_io_config_error(
    cli_name: str,
    invalid_value: str,
    config_dir: Path,
    available: List[str],
) -> str:
    """Format error message for invalid I/O config selection."""
    example = available[0] if available else "json"
    return format_invalid_config_error(
        param_name="--io-cfg",
        config_type=f"{cli_name} I/O",
        config_dir=config_dir,
        available=available,
        example=example,
        invalid_value=invalid_value,
    )


def format_missing_ensemble_config_error(
    config_dir: Path,
    available: List[str],
) -> str:
    """Format error message for missing ensemble config.
    
    Args:
        config_dir: Directory containing ensemble configs
        available: List of available config names
    
    Returns:
        Formatted error message with available options
    """
    example = available[0] if available else "weighted_majority_v1"
    return format_missing_config_error(
        param_name="--ensemble-cfg",
        config_type="ensemble",
        config_dir=config_dir,
        available=available,
        example=example,
    )
