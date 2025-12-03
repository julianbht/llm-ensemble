"""Model configuration schema with centralized nested structure.

Complete configuration for LLM models.
All configuration centralized here - adapters contain no metadata.
"""

from __future__ import annotations
import uuid
from typing import Optional, Literal, Any
from uuid import UUID
from pydantic import BaseModel, Field

from llm_ensemble.infer.schemas.retry_config_schema import RetryConfig
from llm_ensemble.libs.schemas.base_config import BaseConfig


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


class ProviderAdapterConfig(BaseModel):
    """Nested config for provider adapter instantiation details."""

    provider_module: str = Field(
        ...,
        description="Full Python module path to provider adapter"
    )
    provider_class: str = Field(
        ...,
        description="Provider adapter class name in UpperCamelCase"
    )


class ProviderSubConfig(BaseModel):
    """Nested config for provider-specific settings."""

    name_hint: str = Field(
        ...,
        description="Short name hint for this provider config (used for logging/naming)"
    )
    provider_name: Literal["hf", "ollama", "openrouter"] = Field(
        ...,
        description="Provider name (hf, ollama, or openrouter)"
    )
    provider_adapter: ProviderAdapterConfig = Field(
        ...,
        description="Adapter instantiation configuration for provider"
    )

    # Provider-specific fields
    hf_endpoint_url: Optional[str] = Field(
        None,
        description="HF Inference Endpoint URL (HuggingFace only)"
    )
    hf_model_name: Optional[str] = Field(
        None,
        description="HF model repo name (HuggingFace only)"
    )
    openrouter_model_id: Optional[str] = Field(
        None,
        description="OpenRouter model ID (e.g., 'openai/gpt-4') (OpenRouter only)"
    )


class ModelSpecs(BaseModel):
    """Nested config for model inference parameters."""

    name_hint: str = Field(
        ...,
        description="Short name hint for this model spec (used for logging/naming)"
    )
    context_window: int = Field(
        ...,
        gt=0,
        description="Maximum context window size in tokens"
    )

    # Core inference parameters
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


class ModelConfig(BaseConfig):
    """Complete configuration for LLM models.

    All config centralized here - adapters are pure implementation.
    This config includes model identity, provider config, model specs, and pricing.

    Example YAML:
        name_hint: llama-4-maverick-free
        model_name: llama-4-maverick:free
        pricing_info:
            prompt_cost_per_1m_tokens: 0.0
            completion_cost_per_1m_tokens: 0.0
            last_updated: "2025-01-01T00:00:00Z"
        provider_config:
            name_hint: openrouter
            provider_name: openrouter
            provider_adapter:
                provider_module: llm_ensemble.infer.adapters.providers.openrouter_adapter
                provider_class: OpenRouterAdapter
            openrouter_model_id: meta-llama/llama-4-maverick:free
        model_specs:
            name_hint: default
            context_window: 8192
            temperature: null
            max_tokens: null
            ...

    Note: name_hint is inherited from BaseConfig and used for run_name generation.
    """

    id: UUID = Field(
        default_factory=uuid.uuid4,
        description="Random UUID for this model config"
    )

    model_name: str = Field(
        ...,
        description="Model identifier (natural key for Model entity)"
    )

    pricing_info: Optional[PricingInfo] = Field(
        None,
        description="Cost information for this model"
    )

    provider_config: ProviderSubConfig = Field(
        ...,
        description="Provider configuration including adapter"
    )

    model_specs: ModelSpecs = Field(
        ...,
        description="Model inference parameters and capabilities"
    )

    def get_provider(
        self,
        retry_config: RetryConfig,
        logger: Optional[Any] = None,
        api_key: Optional[str] = None,
        timeout: int = 30,
    ) -> Any:
        """Instantiate and return the provider adapter.

        Dynamically imports the provider module and instantiates the provider class.
        Provider name and model name come from config.

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
        kwargs = {
            "provider_name": self.provider_config.provider_name,
            "model_name": self.model_name,
            "retry_config": retry_config,
            "logger": logger,
            "timeout": timeout,
        }
        if api_key is not None:
            kwargs["api_key"] = api_key

        return self._instantiate_adapter(
            self.provider_config.provider_adapter.provider_module,
            self.provider_config.provider_adapter.provider_class,
            **kwargs
        )
