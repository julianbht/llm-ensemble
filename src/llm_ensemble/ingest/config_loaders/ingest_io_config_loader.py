"""Ingest I/O configuration loader.

Loads ingest-specific I/O YAML configurations from configs/io directory.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import yaml

from llm_ensemble.ingest.schemas import IngestIOConfig


def get_default_io_dir() -> Path:
    """Get the default configs/io directory.

    Returns:
        Path to configs/io relative to project root
    """
    # Navigate from this file to project root, then to configs/io
    # This file is at: src/llm_ensemble/ingest/config_loaders/ingest_io_config_loader.py
    # Project root is 4 levels up
    project_root = Path(__file__).parents[4]
    return project_root / "configs" / "io"


def load_ingest_io_config(io_format: str) -> IngestIOConfig:
    """Load an ingest I/O configuration from YAML file.

    Args:
        io_format: I/O format identifier (e.g., "llm_judge_ingest")

    Returns:
        IngestIOConfig object with reader, writer, dataset_id, and data_dir

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If YAML is invalid or missing required fields

    Example:
        >>> config = load_ingest_io_config("llm_judge_ingest")
        >>> config.reader
        'llm_judge_example_reader'
        >>> config.dataset_id
        'llm-judge-2024'
    """
    # Get standard I/O config directory
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

    # Validate and parse into IngestIOConfig
    try:
        return IngestIOConfig(**data)
    except Exception as e:
        raise ValueError(f"Failed to parse ingest I/O config {config_path}: {e}") from e
