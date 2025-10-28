"""Model configuration schema.

Defines the Pydantic schema for LLM model configurations.
Based on OpenRouter API specification for maximum compatibility.
"""

from __future__ import annotations
from typing import Optional, Literal, Any, Union
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Domain model for model configuration (mirrors configs/models/*.yaml).

    Explicit parameters are based on OpenRouter API common parameters.
    Makes frequently-used settings discoverable and type-safe.
    """

    # Identity
    model_id: str = Field(..., description="Model identifier")
    provider: Literal["hf", "ollama", "openrouter"] = Field(..., description="Provider name")

    # Capacity
    context_window: int = Field(..., gt=0, description="Maximum context window size in tokens")

    # Core inference parameters (explicit for discoverability)
    temperature: Optional[float] = Field(
        None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature: 0.0=deterministic, 2.0=very random"
    )
    max_tokens: Optional[int] = Field(
        None,
        gt=0,
        description="Maximum number of tokens to generate"
    )
    top_p: Optional[float] = Field(
        None,
        gt=0.0,
        le=1.0,
        description="Nucleus sampling: only consider tokens with top_p cumulative probability"
    )
    frequency_penalty: Optional[float] = Field(
        None,
        ge=-2.0,
        le=2.0,
        description="Penalize tokens based on frequency in the text so far (-2 to 2)"
    )
    presence_penalty: Optional[float] = Field(
        None,
        ge=-2.0,
        le=2.0,
        description="Penalize tokens based on whether they appear in the text so far (-2 to 2)"
    )
    seed: Optional[int] = Field(
        None,
        description="Random seed for reproducible sampling"
    )
    stop: Optional[list[str]] = Field(
        None,
        description="List of sequences where the API will stop generating further tokens"
    )

    # Output control
    response_format: Optional[dict[str, str]] = Field(
        None,
        description="Force specific output format, e.g., {'type': 'json_object'}"
    )

    # Advanced/provider-specific parameters (catch-all)
    additional_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional provider-specific parameters (e.g., top_k, transforms, etc.)"
    )

    # Capabilities metadata
    capabilities: dict[str, Any] = Field(
        default_factory=dict,
        description="Model capabilities (e.g., multilingual, function_calling, vision)"
    )

    # HuggingFace-specific fields
    hf_endpoint_url: Optional[str] = Field(
        None,
        description="HF Inference Endpoint URL"
    )
    hf_model_name: Optional[str] = Field(
        None,
        description="HF model repo name"
    )

    # OpenRouter-specific fields
    openrouter_model_id: Optional[str] = Field(
        None,
        description="OpenRouter model ID (e.g., 'openai/gpt-4')"
    )
