"""Builder for provider adapters.

Explicit instantiation of provider adapters with provider-specific constructors.
Each provider adapter defines its own constructor signature and configuration needs.

To add a new provider:
1. Create adapter class that extends LLMProvider port
2. Import it here
3. Add explicit instantiation case in create() method
"""

from __future__ import annotations

from llm_ensemble.infer.application.ports.driven.for_invoking_llm import ForInvokingLLM
from llm_ensemble.infer.domain.entities.model_config import ModelConfig
from llm_ensemble.infer.domain.entities.retry_config_schema import RetryConfig
from llm_ensemble.infer.adapters.driven.providers.openrouter_adapter import OpenRouterAdapter
from llm_ensemble.infer.adapters.driven.providers.ollama_adapter import OllamaAdapter
from llm_ensemble.infer.adapters.driven.providers.mock_adapter import MockLLMAdapter


class ProviderFactory:
    """Builder for creating provider adapter instances."""

    @staticmethod
    def create(
        provider_name: str,
        model_config: ModelConfig,
        retry_config: RetryConfig,
    ) -> ForInvokingLLM:
        """Build and return a provider adapter instance.

        Uses explicit instantiation per provider to allow provider-specific
        constructor signatures and configuration.

        Args:
            provider_name: Name of the provider (e.g., 'openrouter', 'ollama')
            model_config: Complete model configuration (model_id, temperature, max_tokens, etc.)
            retry_config: Retry configuration for exponential backoff

        Returns:
            Instantiated provider adapter

        Raises:
            ValueError: If provider not found
        """
        if provider_name == "openrouter":
            return OpenRouterAdapter(
                model_config=model_config,
                retry_config=retry_config,
            )
        elif provider_name == "ollama":
            return OllamaAdapter(
                model_config=model_config,
                retry_config=retry_config,
            )
        elif provider_name == "mock":
            return MockLLMAdapter(
                model_config=model_config,
                retry_config=retry_config,
            )
        else:
            available = ", ".join(sorted(["openrouter", "ollama", "mock"]))
            raise ValueError(
                f"Provider '{provider_name}' not found. "
                f"Available: {available}"
            )

    @staticmethod
    def list_available() -> list[str]:
        """List all available provider names.

        Returns:
            Sorted list of provider names
        """
        return sorted(["openrouter", "ollama", "mock"])

    @staticmethod
    def has_provider(provider_name: str) -> bool:
        """Check if provider is available.

        Args:
            provider_name: Name of the provider

        Returns:
            True if provider exists
        """
        return provider_name in ["openrouter", "ollama", "mock"]
