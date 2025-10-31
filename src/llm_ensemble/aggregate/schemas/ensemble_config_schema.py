"""Ensemble configuration schema."""

from __future__ import annotations
from typing import Optional
from pydantic import Field

from llm_ensemble.libs.schemas.base_config import BaseConfig


class EnsembleConfig(BaseConfig):
    """Configuration for ensemble aggregation strategies.
    
    Specifies which strategy to use. Keep it simple - just the strategy name.
    
    Example YAML:
        strategy: majority_vote
    """
    
    strategy: str = Field(
        ...,
        description=(
            "Name of the aggregation strategy to use. "
            "Supported: 'majority_vote'"
        )
    )
    
    # Optional name hint for run_id generation (derived from filename by loader)
    name_hint: Optional[str] = Field(
        default=None,
        description="Short name hint for run_id generation (e.g., 'majority_vote')"
    )
