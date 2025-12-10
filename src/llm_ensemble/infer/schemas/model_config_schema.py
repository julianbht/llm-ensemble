"""Model configuration schema.

Flat configuration for LLM models matching the database ORM structure.
"""

from __future__ import annotations
import uuid
from typing import Optional, Any, Dict
from uuid import UUID
from pydantic import Field

from llm_ensemble.libs.schemas.base_config import BaseConfig


class ModelConfig(BaseConfig):
    """Complete configuration for LLM models.

    Flat structure matching ModelConfigORM for clean mapping.

    Example YAML:
        name_hint: gpt-4-turbo
        model_id: gpt-4-turbo-2024-04-09
        context_window: 128000
        temperature: 0.7
        max_tokens: 4096
        capabilities:
            multilingual: true
            function_calling: true
        additional_params:
            stop: ["END"]
            response_format: {"type": "json_object"}
    """

    id: UUID = Field(
        default_factory=uuid.uuid4,
        description="Random UUID for this model config"
    )

    # Model identity
    model_id: str = Field(
        ...,
        description="Model identifier (e.g., 'gpt-4', 'llama-3-70b', 'meta-llama/llama-4-maverick:free')"
    )

    # Model capabilities
    context_window: int = Field(
        ...,
        gt=0,
        description="Maximum context window size in tokens"
    )

    capabilities: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Model capabilities (e.g., multilingual, function_calling, vision)"
    )

    # Inference parameters
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature: 0.0=deterministic, 2.0=very random"
    )

    max_tokens: Optional[int] = Field(
        default=None,
        gt=0,
        description="Maximum number of tokens to generate"
    )

    top_p: Optional[float] = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description="Nucleus sampling: only consider tokens with top_p cumulative probability"
    )

    frequency_penalty: Optional[float] = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        description="Penalize tokens based on frequency in the text so far (-2 to 2)"
    )

    presence_penalty: Optional[float] = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        description="Penalize tokens based on whether they appear in the text so far (-2 to 2)"
    )

    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducible sampling"
    )

    # Additional parameters as catch-all (stop sequences, response_format, etc.)
    additional_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional provider-specific parameters (e.g., stop, response_format, top_k)"
    )
