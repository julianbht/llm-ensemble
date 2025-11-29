"""Model configuration schema.

Defines the Pydantic schema for LLM model configurations.
Based on OpenRouter API specification for maximum compatibility.
"""

from __future__ import annotations
from typing import Optional, Literal, Any
from uuid import UUID
from pydantic import BaseModel, Field

from llm_ensemble.infer.schemas.retry_config_schema import RetryConfig
from llm_ensemble.libs.schemas.base_config import BaseConfig
from llm_ensemble.libs.db import compute_model_config_uuid


class PricingInfo(BaseModel):
    """Pricing information for LLM model (cost per 1M tokens)."""

    prompt_cost_per_1m_tokens: float = Field(
        ...,
        ge=0.0,
        description="Cost in USD per 1 million prompt tokens"
    )
    completion_cost_per_1m_tokens: float = Field(
        ...,
        ge=0.0,
        description="Cost in USD per 1 million completion tokens"
    )
    last_updated: str = Field(
        ...,
        description="ISO 8601 timestamp of when pricing was last updated"
    )


class ModelConfig(BaseConfig):
    """Configuration entity for LLM models.

    Config is persisted as ModelConfigORM with deterministic UUID.
    Explicit parameters are based on OpenRouter API common parameters.
    Makes frequently-used settings discoverable and type-safe.
    """

    id: UUID = Field(
        ...,
        description="Deterministic UUID computed from config name (natural key)"
    )

    # Identity
    model_id: str = Field(..., description="Model identifier")
    provider: Literal["hf", "ollama", "openrouter"] = Field(..., description="Provider name")

    # Dynamic adapter loading
    provider_module: str = Field(..., description="Full Python module path to provider adapter (e.g., 'llm_ensemble.infer.adapters.providers.openrouter_adapter')")
    provider_class: str = Field(..., description="Provider adapter class name in UpperCamelCase (e.g., 'OpenRouterAdapter')")

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

    # Pricing information
    pricing: Optional[PricingInfo] = Field(
        None,
        description="Cost information for this model (typically auto-updated via update_model_pricing.py)"
    )

    @classmethod
    def create(cls, name: str, **kwargs) -> "ModelConfig":
        """Create ModelConfig with computed ID from name.

        Args:
            name: Config name (natural key, typically from filename)
            **kwargs: Other config fields (model_id, provider, etc.)

        Returns:
            ModelConfig with computed ID
        """
        config_id = compute_model_config_uuid(name)
        return cls(id=config_id, name=name, **kwargs)

    def get_provider(
        self,
        retry_config: RetryConfig,
        logger: Optional[Any] = None,
        api_key: Optional[str] = None,
        timeout: int = 30,
    ) -> Any:
        """Instantiate and return the provider adapter.

        Dynamically imports the provider module and instantiates the provider class.

        Args:
            retry_config: Retry configuration for exponential backoff
            logger: Optional logger for retry events
            api_key: Optional API key (if not provided, adapter will use env vars)
            timeout: Request timeout in seconds (default: 30)

        Returns:
            Instance of the provider adapter (LLMProvider)

        Raises:
            ImportError: If the provider module cannot be imported
            AttributeError: If the provider class doesn't exist in the module
        """
        # Build kwargs based on what the provider needs
        # Note: retry_config and logger are REQUIRED for all providers (base class requires them)
        kwargs = {
            "retry_config": retry_config,
            "logger": logger,
            "timeout": timeout,
        }
        if api_key is not None:
            kwargs["api_key"] = api_key

        return self._instantiate_adapter(self.provider_module, self.provider_class, **kwargs)
