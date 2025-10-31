"""Configuration loaders for aggregate CLI."""

from __future__ import annotations
from pathlib import Path
import yaml

from llm_ensemble.aggregate.schemas.ensemble_config_schema import EnsembleConfig
from llm_ensemble.libs.runtime.path_manager import PathManager


def load_ensemble_config(config_name: str) -> EnsembleConfig:
    """Load ensemble configuration from YAML file.
    
    Args:
        config_name: Name of the ensemble config (without .yaml extension)
        
    Returns:
        Parsed and validated EnsembleConfig object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
        
    Example:
        >>> config = load_ensemble_config("majority_vote")
        >>> config.strategy
        'majority_vote'
    """
    # Get ensembles config directory
    config_dir = PathManager.get_project_root() / "configs" / "ensembles"
    config_path = config_dir / f"{config_name}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Ensemble config not found: {config_path}\n"
            f"Available configs in {config_dir.relative_to(PathManager.get_project_root())}:\n"
            f"{', '.join(p.stem for p in config_dir.glob('*.yaml'))}"
        )
    
    # Load and parse YAML
    with config_path.open("r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
    
    # Add name_hint from filename for run_id generation
    config_data["name_hint"] = config_name
    
    # Validate with Pydantic
    return EnsembleConfig(**config_data)
