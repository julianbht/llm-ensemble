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


def load_dataset_config(config_name: str, config_dir: Optional[Path] = None) -> DatasetConfig:
    """Load dataset configuration by config filename (without extension).

    Args:
        config_name: Config filename without extension (e.g., 'llm_judge_challenge')
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

    # Try to find config file by name (try .yaml first, then .yml)
    config_file = config_dir / f"{config_name}.yaml"
    if not config_file.exists():
        config_file = config_dir / f"{config_name}.yml"

    if not config_file.exists():
        # List available configs for helpful error message
        available = [f.stem for f in config_dir.glob("*.yaml")] + [f.stem for f in config_dir.glob("*.yml")]
        raise FileNotFoundError(
            f"No config found for config_name='{config_name}'. "
            f"Available configs: {sorted(set(available))}"
        )

    # Load and return the config
    try:
        return DatasetConfig.from_yaml(config_file)
    except (ValueError, KeyError, yaml.YAMLError) as e:
        raise ValueError(f"Invalid config file {config_file}: {e}")
