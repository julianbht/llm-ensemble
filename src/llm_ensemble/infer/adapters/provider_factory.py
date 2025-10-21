"""Factory for creating LLM provider instances.

Maps provider names from model configs to concrete adapter implementations,
enabling dependency injection and loose coupling.

All providers receive builder, parser, and template as constructor dependencies,
following the dependency injection principle.
"""

from __future__ import annotations
from typing import Optional
from jinja2 import Template

from llm_ensemble.infer.schemas import ModelConfig
from llm_ensemble.infer.ports import LLMProvider, PromptBuilder, ResponseParser
from llm_ensemble.infer.adapters.providers import (
    OpenRouterAdapter,
    OllamaAdapter,
    HuggingFaceAdapter,
)


def get_provider(
    model_config: ModelConfig,
    builder: PromptBuilder,
    parser: ResponseParser,
    template: Template,
    api_key: Optional[str] = None,
    timeout: int = 30,
) -> LLMProvider:
    """Create and return the appropriate LLM provider adapter.

    Factory function that instantiates the correct provider implementation
    based on the model configuration's provider field, injecting all necessary
    dependencies (builder, parser, template).

    Args:
        model_config: Model configuration with provider specification
        builder: PromptBuilder adapter instance
        parser: ResponseParser adapter instance
        template: Jinja2 Template instance for prompt formatting
        api_key: Optional API key (if not provided, will use env vars)
        timeout: Request timeout in seconds (default: 30)

    Returns:
        LLMProvider instance (OpenRouterAdapter, OllamaAdapter, etc.)

    Raises:
        ValueError: If provider is not supported

    Example:
        >>> from llm_ensemble.infer.config_loaders import load_model_config
        >>> from llm_ensemble.infer.prompt_builders import load_builder
        >>> from llm_ensemble.infer.response_parsers import load_parser
        >>> 
        >>> config = load_model_config("gpt-oss-20b")
        >>> builder = load_builder("thomas")
        >>> parser = load_parser("thomas")
        >>> template = load_template("thomas-et-al-prompt.jinja")
        >>> 
        >>> provider = get_provider(config, builder, parser, template)
        >>> isinstance(provider, OpenRouterAdapter)
        True
    """
    provider_name = model_config.provider.lower()

    if provider_name == "openrouter":
        return OpenRouterAdapter(
            builder=builder,
            parser=parser,
            template=template,
            api_key=api_key,
            timeout=timeout,
        )
    elif provider_name == "ollama":
        return OllamaAdapter(
            builder=builder,
            parser=parser,
            template=template,
            timeout=timeout,
        )
    elif provider_name == "hf" or provider_name == "huggingface":
        return HuggingFaceAdapter(
            builder=builder,
            parser=parser,
            template=template,
            api_token=api_key,
            timeout=timeout,
        )
    else:
        raise ValueError(
            f"Unsupported provider: {model_config.provider}. "
            f"Supported providers: openrouter, ollama, hf"
        )
