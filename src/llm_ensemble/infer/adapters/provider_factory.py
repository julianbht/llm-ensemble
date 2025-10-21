"""Factory for creating LLM provider instances.

Maps provider names from model configs to concrete adapter implementations,
enabling dependency injection and loose coupling.
"""

from __future__ import annotations
from typing import Optional

from llm_ensemble.infer.schemas import ModelConfig
from llm_ensemble.infer.ports import LLMProvider
from llm_ensemble.infer.adapters.providers import (
    OpenRouterAdapter,
    OllamaAdapter,
    HuggingFaceAdapter,
)


def get_provider(
    model_config: ModelConfig,
    api_key: Optional[str] = None,
    timeout: int = 30,
) -> LLMProvider:
    """Create and return the appropriate LLM provider adapter.

    Factory function that instantiates the correct provider implementation
    based on the model configuration's provider field.

    Args:
        model_config: Model configuration with provider specification
        api_key: Optional API key (if not provided, will use env vars)
        timeout: Request timeout in seconds (default: 30)

    Returns:
        LLMProvider instance (OpenRouterAdapter, OllamaAdapter, etc.)

    Raises:
        ValueError: If provider is not supported

    Example:
        >>> from llm_ensemble.infer.config_loaders import load_model_config
        >>> config = load_model_config("gpt-oss-20b")
        >>> provider = get_provider(config)
        >>> isinstance(provider, OpenRouterAdapter)
        True
    """
    provider_name = model_config.provider.lower()

    if provider_name == "openrouter":
        return OpenRouterAdapter(api_key=api_key, timeout=timeout)
    elif provider_name == "ollama":
        return OllamaAdapter(timeout=timeout)
    elif provider_name == "hf" or provider_name == "huggingface":
        return HuggingFaceAdapter(api_token=api_key, timeout=timeout)
    else:
        raise ValueError(
            f"Unsupported provider: {model_config.provider}. "
            f"Supported providers: openrouter, ollama, hf"
        )
