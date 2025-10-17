"""Model configuration schema.

Defines the Pydantic schema for LLM model configurations.
"""

from __future__ import annotations
from typing import Optional, Literal, Any
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Domain model for model configuration (mirrors configs/models/*.yaml)."""

    model_id: str
    provider: Literal["hf", "ollama", "openrouter"]
    context_window: int
    default_params: dict[str, Any] = Field(default_factory=dict)
    capabilities: dict[str, Any] = Field(default_factory=dict)

    # HuggingFace-specific fields
    hf_endpoint_url: Optional[str] = Field(None, description="HF Inference Endpoint URL")
    hf_model_name: Optional[str] = Field(None, description="HF model repo name")

    # OpenRouter-specific fields
    openrouter_model_id: Optional[str] = Field(None, description="OpenRouter model ID")
