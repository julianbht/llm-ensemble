from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for a dataset, loaded from configs/datasets/*.yaml"""

    dataset_id: str
    adapter: str
    data_dir: Path
    files: Dict[str, str]
    description: Optional[str] = None
    version: Optional[str] = None

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> DatasetConfig:
        """Load dataset config from YAML file."""
        with yaml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            raise ValueError(f"Empty config file: {yaml_path}")

        # Validate required fields
        required = {"dataset_id", "adapter", "data_dir", "files"}
        missing = required - set(data.keys())
        if missing:
            raise ValueError(f"Missing required fields in {yaml_path}: {missing}")

        return cls(
            dataset_id=data["dataset_id"],
            adapter=data["adapter"],
            data_dir=Path(data["data_dir"]),
            files=data["files"],
            description=data.get("description"),
            version=data.get("version"),
        )


def load_dataset_config(dataset_id: str, config_dir: Optional[Path] = None) -> DatasetConfig:
    """Load dataset configuration by dataset_id.

    Args:
        dataset_id: Dataset identifier (e.g., 'llm-judge-2024')
        config_dir: Optional config directory override (defaults to PROJECT_ROOT/configs/datasets)

    Returns:
        DatasetConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    if config_dir is None:
        # Default to PROJECT_ROOT/configs/datasets
        project_root = Path(__file__).parent.parent.parent.parent.parent
        config_dir = project_root / "configs" / "datasets"

    # Look for matching YAML file
    # Try exact match first, then check all files for matching dataset_id
    config_files = list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml"))

    for config_file in config_files:
        try:
            config = DatasetConfig.from_yaml(config_file)
            if config.dataset_id == dataset_id:
                return config
        except (ValueError, KeyError, yaml.YAMLError):
            # Skip invalid configs
            continue

    # Not found
    available = [f.stem for f in config_files]
    raise FileNotFoundError(
        f"No config found for dataset_id='{dataset_id}'. "
        f"Available configs: {available}"
    )
