"""Model configuration factory.

Factory for loading YAML model configuration files and returning ModelConfig domain objects.
"""

from __future__ import annotations

from llm_ensemble.infer.schemas.model_config_schema import ModelConfig
from llm_ensemble.libs.config import load_yaml_config
from llm_ensemble.libs.runtime.path_manager import PathManager


class ModelConfigFactory:
    """Factory for loading model configurations from YAML files."""

    @staticmethod
    def load(model_id: str) -> ModelConfig:
        """Load a model configuration from YAML file.

        Args:
            model_id: Model identifier (e.g., "phi3-mini", "gpt-oss-20b")

        Returns:
            ModelConfig object with all settings loaded from YAML

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If YAML is invalid or missing required fields
        """
        return load_yaml_config(
            config_name=model_id,
            config_dir=PathManager.get_model_configs_dir(),
            schema=ModelConfig,
            config_type="model",
        )
