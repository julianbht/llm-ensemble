"""Shared I/O configuration loader.

Loads I/O YAML configurations from the centralized configs/io directory.
These configs bundle reader and writer adapters for specific formats (ndjson, parquet, etc.).

This is a shared loader used by all CLIs.
"""

from __future__ import annotations

from llm_ensemble.libs.schemas import IOConfig
from llm_ensemble.libs.config.yaml_config_loader import load_yaml_config
from llm_ensemble.libs.runtime.path_manager import PathManager


def load_io_config(io_format: str) -> IOConfig:
    """Load an I/O configuration from YAML file.

    Args:
        io_format: I/O format identifier (e.g., "ndjson", "parquet", "llm_judge_ingest")

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
    return load_yaml_config(
        config_name=io_format,
        config_dir=PathManager.get_io_configs_dir(),
        schema=IOConfig,
        config_type="I/O",
    )
