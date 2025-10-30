"""Ingest I/O configuration loader.

Loads ingest-specific I/O YAML configurations from configs/io directory.
"""

from __future__ import annotations

from llm_ensemble.ingest.schemas import IngestIOConfig
from llm_ensemble.libs.config import load_yaml_config
from llm_ensemble.libs.runtime.path_manager import PathManager


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
        >>> config = load_ingest_io_config("llm_judge_challenge")
        >>> config.reader
        'llm_judge_sample_reader'
        >>> config.dataset_id
        'llm-judge-challenge-2024'
    """
    return load_yaml_config(
        config_name=io_format,
        config_dir=PathManager.get_io_configs_dir(),
        schema=IngestIOConfig,
        config_type="ingest I/O",
    )
