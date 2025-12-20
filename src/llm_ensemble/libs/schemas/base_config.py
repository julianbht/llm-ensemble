"""Base configuration schema.

Defines the base Pydantic schema for all configuration types.
Provides common fields (id, name).

Configuration loading is handled by the startup layer, not by the schemas themselves.
"""

from __future__ import annotations
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


class BaseConfig(BaseModel):
    """Base configuration class for all configs.

    All configuration types (ModelConfig, RetryConfig, IOConfig, LoggingConfig)
    inherit from this base class to get consistent fields:
    - id: Random UUID for database identity
    - name: Configuration name from filename
    """

    id: UUID = Field(
        default_factory=uuid4,
        description="Random UUID identifier for this config"
    )

    name: str = Field(
        ...,
        description=(
            "Configuration name, derived from filename (without extension). "
            "Injected during loading. Example: 'gpt-oss-20b' from 'gpt-oss-20b.yaml'"
        ),
    )
