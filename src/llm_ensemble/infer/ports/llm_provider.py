"""Port interface for LLM inference providers.

Defines the abstract contract that all LLM provider adapters must implement.
This allows the orchestrator to depend on an abstraction rather than concrete
provider implementations (OpenRouter, Ollama, HuggingFace, etc.).
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

    Providers are initialized with injected PromptBuilder and ResponseParser
    ports, following dependency inversion principles.

    Example:
        >>> class OpenRouterAdapter(LLMProvider):
        ...     def __init__(self, prompt_builder, response_parser, ...):
        ...         self.prompt_builder = prompt_builder
        ...         self.response_parser = response_parser
        ...     def infer(self, examples, model_config):
        ...         # Implementation using injected ports
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
