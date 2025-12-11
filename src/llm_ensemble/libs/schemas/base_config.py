"""Base configuration schema.

Defines the base Pydantic schema for all configuration types.
Provides common fields (id, name, name_hint) and a load() classmethod.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


class BaseConfig(BaseModel):
    """Base configuration class for all configs.

    All configuration types (ModelConfig, RetryConfig, IOConfig, LoggingConfig)
    inherit from this base class to get consistent fields:
    - id: Random UUID for database identity
    - name: Configuration name from filename (injected by load())
    - name_hint: Optional short hint for run ID generation (from YAML)
    """

    id: UUID = Field(
        default_factory=uuid4,
        description="Random UUID identifier for this config"
    )

    name: str = Field(
        ...,
        description=(
            "Configuration name, derived from filename (without extension). "
            "This field is injected by load() and should NOT be set in YAML files. "
            "Example: 'gpt-oss-20b' from 'gpt-oss-20b.yaml'"
        ),
    )

    name_hint: Optional[str] = Field(
        None,
        description=(
            "Short name hint for run ID generation (e.g., 'gpt20b', 'thomas', 'llmjudge'). "
            "Specified in YAML files. If not provided, this config won't contribute to the run ID. "
            "Keep it short (5-15 chars) and use only alphanumeric characters, hyphens, or underscores."
        ),
    )

    @classmethod
    def load(cls, config_name: str, config_dir: Path) -> "BaseConfig":
        """Load configuration from YAML file and inject name field.

        Args:
            config_name: Name of config file (without .yaml extension)
            config_dir: Directory containing the config file

        Returns:
            Config instance with name injected from filename

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If YAML is invalid or missing required fields
        """
        from llm_ensemble.libs.config import load_yaml_config

        return load_yaml_config(
            config_name=config_name,
            config_dir=config_dir,
            schema=cls,
            config_type=cls.__name__,
        )
