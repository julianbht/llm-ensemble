"""Port interface for LLM inference providers.

Defines the abstract contract that all LLM provider adapters must implement.
This allows the orchestrator to depend on an abstraction rather than concrete
provider implementations (OpenRouter, Ollama, HuggingFace, etc.).

Providers receive all dependencies (builder, parser, template) via dependency injection,
eliminating the need for internal config loading.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterator

from llm_ensemble.ingest.schemas import JudgingExample
from llm_ensemble.infer.schemas import ModelJudgement, ModelConfig


class LLMProvider(ABC):
    """Abstract base class for LLM inference providers.

    All provider adapters (OpenRouter, Ollama, HuggingFace) must inherit
    from this class and implement the infer() method.

    Providers receive PromptBuilder, ResponseParser, and Template as constructor
    dependencies, configured by the orchestrator. Providers only implement
    the provider-specific API communication logic.

    Example:
        >>> class OpenRouterAdapter(LLMProvider):
        ...     def __init__(self, builder, parser, template, api_key):
        ...         self.builder = builder
        ...         self.parser = parser
        ...         self.template = template
        ...     
        ...     def infer(self, examples, model_config):
        ...         # Implementation using injected dependencies
        ...         yield judgement
    """

    @abstractmethod
    def infer(
        self,
        examples: Iterator[JudgingExample],
        model_config: ModelConfig,
    ) -> Iterator[ModelJudgement]:
        """Run inference on examples and yield judgements.

        Args:
            examples: Iterator of JudgingExample objects to judge
            model_config: Model configuration with provider and settings

        Yields:
            ModelJudgement objects with predictions and metadata

        Raises:
            ValueError: If configuration is invalid
            Exception: If provider API fails
        """
        pass
