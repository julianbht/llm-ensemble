"""Factory for creating LLM provider instances.

Maps provider names from model configs to concrete adapter implementations,
enabling dependency injection and loose coupling.
"""

from __future__ import annotations
from typing import Optional

from llm_ensemble.infer.schemas import ModelConfig
from llm_ensemble.infer.ports import LLMProvider, PromptBuilder
from llm_ensemble.infer.adapters.providers import (
    OpenRouterAdapter,
    OllamaAdapter,
    HuggingFaceAdapter,
)


def get_provider(
    model_config: ModelConfig,
    prompt_builder: PromptBuilder,
    api_key: Optional[str] = None,
    timeout: int = 30,
) -> LLMProvider:
    """Create and return the appropriate LLM provider adapter.

    Factory function that instantiates the correct provider implementation
    based on the model configuration's provider field.

    Providers return RAW LLMResponse objects (unparsed text + metadata).
    The InferenceService coordinates parsing via ResponseParser.

    Args:
        model_config: Model configuration with provider specification
        prompt_builder: PromptBuilder port for building prompts
        api_key: Optional API key (if not provided, will use env vars)
        timeout: Request timeout in seconds (default: 30)

    Returns:
        LLMProvider instance (OpenRouterAdapter, OllamaAdapter, etc.)

    Raises:
        ValueError: If provider is not supported

    Example:
        >>> from llm_ensemble.infer.config_loaders import load_model_config, load_prompt_config
        >>> from llm_ensemble.infer.adapters.prompt_builder_factory import get_prompt_builder
        >>> model_config = load_model_config("gpt-oss-20b")
        >>> prompt_config = load_prompt_config("thomas-et-al-prompt")
        >>> builder = get_prompt_builder(prompt_config)
        >>> provider = get_provider(model_config, builder)
        >>> isinstance(provider, OpenRouterAdapter)
        True
    """
    provider_name = model_config.provider.lower()

    if provider_name == "openrouter":
        return OpenRouterAdapter(
            prompt_builder=prompt_builder,
            api_key=api_key,
            timeout=timeout,
        )
    elif provider_name == "ollama":
        return OllamaAdapter(
            prompt_builder=prompt_builder,
            timeout=timeout,
        )
    elif provider_name == "hf" or provider_name == "huggingface":
        return HuggingFaceAdapter(
            prompt_builder=prompt_builder,
            api_token=api_key,
            timeout=timeout,
        )
    else:
        raise ValueError(
            f"Unsupported provider: {model_config.provider}. "
            f"Supported providers: openrouter, ollama, hf"
        )
