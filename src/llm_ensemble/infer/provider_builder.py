"""Builder for provider adapters.

Simple, explicit mapping of provider names to adapter classes.
No decorators, no hidden registration - just a clear dictionary.

To add a new provider:
1. Create adapter class that extends LLMProvider port
2. Import it here
3. Add to PROVIDERS dict
"""

from __future__ import annotations
from typing import Dict, Type

from llm_ensemble.infer.ports import LLMProvider
from llm_ensemble.infer.schemas.retry_config_schema import RetryConfig
from llm_ensemble.infer.schemas.model_config_schema import ModelConfig
from llm_ensemble.infer.adapters.providers.openrouter_adapter import OpenRouterAdapter
from llm_ensemble.infer.adapters.providers.ollama_adapter import OllamaAdapter


# Explicit mapping of provider names to adapter classes
PROVIDERS: Dict[str, Type[LLMProvider]] = {
    "openrouter": OpenRouterAdapter,
    "ollama": OllamaAdapter,
}


class ProviderAdapterBuilder:
    """Builder for creating provider adapter instances."""

    @staticmethod
    def build(
        provider_name: str,
        model_config: ModelConfig,
        retry_config: RetryConfig,
    ) -> LLMProvider:
        """Build and return a provider adapter instance.

        Args:
            provider_name: Name of the provider (e.g., 'openrouter', 'ollama')
            model_config: Model configuration with provider-specific settings
            retry_config: Retry configuration for exponential backoff

        Returns:
            Instantiated provider adapter

        Raises:
            ValueError: If provider not found
        """
        if provider_name not in PROVIDERS:
            available = ", ".join(sorted(PROVIDERS.keys()))
            raise ValueError(
                f"Provider '{provider_name}' not found. "
                f"Available: {available}"
            )

        adapter_class = PROVIDERS[provider_name]

        # Build provider-specific kwargs
        kwargs = {
            "provider_name": provider_name,
            "model_name": model_config.model_name,
            "retry_config": retry_config,
        }

        # Add provider-specific parameters based on provider type
        if provider_name == "openrouter":
            # OpenRouterAdapter-specific parameters (api_key comes from env)
            kwargs["timeout"] = 30
        elif provider_name == "ollama":
            # OllamaAdapter-specific parameters
            kwargs["base_url"] = "http://localhost:11434"
            kwargs["timeout"] = 60

        return adapter_class(**kwargs)

    @staticmethod
    def list_available() -> list[str]:
        """List all available provider names.

        Returns:
            Sorted list of provider names
        """
        return sorted(PROVIDERS.keys())

    @staticmethod
    def has_provider(provider_name: str) -> bool:
        """Check if provider is available.

        Args:
            provider_name: Name of the provider

        Returns:
            True if provider exists
        """
        return provider_name in PROVIDERS
