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

from llm_ensemble.infer.application.ports.llm_provider_port import LLMProviderPort
from llm_ensemble.infer.domain.entities.model_config import ModelConfig
from llm_ensemble.infer.adapters.providers.openrouter_adapter import OpenRouterAdapter
from llm_ensemble.infer.adapters.providers.ollama_adapter import OllamaAdapter


# Explicit mapping of provider names to adapter classes
PROVIDERS: Dict[str, Type[LLMProviderPort]] = {
    "openrouter": OpenRouterAdapter,
    "ollama": OllamaAdapter,
}


class ProviderFactory:
    """Builder for creating provider adapter instances."""

    @staticmethod
    def create(
        provider_name: str,
        model_config: ModelConfig,
    ) -> LLMProviderPort:
        """Build and return a provider adapter instance.

        Adapters are configured once with the full model configuration.
        Retry logic is handled by a separate wrapper.

        Args:
            provider_name: Name of the provider (e.g., 'openrouter', 'ollama')
            model_config: Complete model configuration (model_id, temperature, max_tokens, etc.)

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

        # Pass full model config to provider (configured once at initialization)
        return adapter_class(
            provider_name=provider_name,
            model_config=model_config,
        )

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
