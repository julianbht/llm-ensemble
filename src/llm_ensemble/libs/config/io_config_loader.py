"""Shared I/O configuration loader.

Loads I/O YAML configurations from CLI-specific configs/io/{cli_name}/ directories.
These configs bundle reader and writer adapters for specific formats (json, parquet, etc.).

This is a shared loader used by all CLIs.
"""

from __future__ import annotations

from llm_ensemble.libs.schemas import IOConfig
from llm_ensemble.libs.config.yaml_config_loader import load_yaml_config
from llm_ensemble.libs.runtime.path_manager import PathManager


def load_io_config(io_format: str, cli_name: str) -> IOConfig:
    """Load an I/O configuration from YAML file.

    Args:
        io_format: I/O format identifier (e.g., "json", "llm_judge_json")
        cli_name: CLI name (e.g., "ingest", "infer", "aggregate", "evaluate")

    Returns:
        IOConfig object with reader and writer adapter specifications

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If YAML is invalid or missing required fields

    Example:
        >>> config = load_io_config("json", "infer")
        >>> config.reader_module
        'llm_ensemble.infer.adapters.io.fully_populated_json_reader'
    """
    return load_yaml_config(
        config_name=io_format,
        config_dir=PathManager.get_io_configs_dir(cli_name),
        schema=IOConfig,
        config_type="I/O",
    )
