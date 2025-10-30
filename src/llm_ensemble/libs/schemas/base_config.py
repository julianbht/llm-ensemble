"""Base configuration schema.

Defines the base Pydantic schema for all configuration types that participate
in run ID generation. This ensures consistency across model, prompt, I/O, and
ensemble configurations.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class BaseConfig(BaseModel):
    """Base configuration class for all configs that participate in run ID generation.

    All configuration types (ModelConfig, PromptConfig, IOConfig, EnsembleConfig)
    should inherit from this base class to ensure consistent metadata fields.

    The name_hint field allows configs to explicitly contribute to human-readable
    run IDs, making it easy to identify what configurations were used in a run.
    """

    name_hint: Optional[str] = Field(
        None,
        description=(
            "Short name hint for run ID generation (e.g., 'gpt20b', 'thomas', 'llmjudge'). "
            "If not provided, this config won't contribute to the run ID. "
            "Keep it short (5-15 chars) and use only alphanumeric characters, hyphens, or underscores."
        ),
    )
