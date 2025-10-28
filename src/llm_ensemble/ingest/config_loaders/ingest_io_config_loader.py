"""Ingest I/O configuration loader.

Loads ingest-specific I/O YAML configurations from configs/io directory.
"""

from __future__ import annotations
import yaml

from llm_ensemble.ingest.schemas import IngestIOConfig
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
    # Get standard I/O config directory using PathManager
    io_dir = PathManager.get_io_configs_dir()

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
