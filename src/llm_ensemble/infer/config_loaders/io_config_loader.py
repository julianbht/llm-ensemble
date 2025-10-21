"""I/O configuration loader.

Loads I/O YAML configurations from the centralized configs/io directory.
These configs bundle reader and writer adapters for specific formats (ndjson, parquet, etc.).
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import yaml

from llm_ensemble.infer.schemas.io_config_schema import IOConfig


def get_default_io_dir() -> Path:
    """Get the default configs/io directory.

    Returns:
        Path to configs/io relative to project root
    """
    # Navigate from this file to project root, then to configs/io
    # This file is at: src/llm_ensemble/infer/config_loaders/io_config_loader.py
    # Project root is 4 levels up
    project_root = Path(__file__).parents[4]
    return project_root / "configs" / "io"


def load_io_config(
    io_format: str,
    io_dir: Optional[Path] = None,
) -> IOConfig:
    """Load an I/O configuration from YAML file.

    Args:
        io_format: I/O format identifier (e.g., "ndjson", "parquet")
        io_dir: Directory containing I/O configs (defaults to configs/io)

    Returns:
        IOConfig object with reader and writer adapter names

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If YAML is invalid or missing required fields

    Example:
        >>> config = load_io_config("ndjson")
        >>> config.reader
        'ndjson_example_reader'
        >>> config.writer
        'ndjson_judgement_writer'
    """
    # Determine I/O directory
    if io_dir is None:
        io_dir = get_default_io_dir()

    # Build path to config file
    config_path = io_dir / f"{io_format}.yaml"

    if not config_path.exists():
        available = [p.stem for p in io_dir.glob("*.yaml")] if io_dir.exists() else []
        raise FileNotFoundError(
            f"I/O config not found: {config_path}\n"
            f"Available I/O formats in {io_dir}:\n"
            + "\n".join(f"  - {fmt}" for fmt in available)
        )

    # Load YAML
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid config file {config_path}: expected YAML object")

    # Validate and parse into IOConfig
    try:
        return IOConfig(**data)
    except Exception as e:
        raise ValueError(f"Failed to parse I/O config {config_path}: {e}") from e
