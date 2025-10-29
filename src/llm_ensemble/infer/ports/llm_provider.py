"""Port interface for LLM inference providers.

Defines the abstract contract that all LLM provider adapters must implement.
This allows the orchestrator to depend on an abstraction rather than concrete
provider implementations (OpenRouter, Ollama, HuggingFace, etc.).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterator

from llm_ensemble.ingest.schemas import JudgingSample
from llm_ensemble.infer.schemas.llm_response import LLMResponse
from llm_ensemble.infer.schemas import ModelConfig


class LLMProvider(ABC):
    """Abstract base class for LLM inference providers.

    All provider adapters (OpenRouter, Ollama, HuggingFace) must inherit
    from this class and implement the infer() method.

    Providers are initialized with injected PromptBuilder and ResponseParser
    ports, following dependency inversion principles.

    Providers yield (sample, LLMResponse) tuples - they do NOT attach manifests.
    The domain service handles manifest attachment to create final LLMJudgement objects.

    Example:
        >>> class OpenRouterAdapter(LLMProvider):
        ...     def __init__(self, prompt_builder, response_parser, ...):
        ...         self.prompt_builder = prompt_builder
        ...         self.response_parser = response_parser
        ...     def infer(self, samples, model_config):
        ...         for sample in samples:
        ...             response = self._call_llm(sample)
        ...             yield (sample, response)
    """

    @abstractmethod
    def infer(
        self,
        samples: Iterator[JudgingSample],
        model_config: ModelConfig,
    ) -> Iterator[tuple[JudgingSample, LLMResponse]]:
        """Run inference on samples and yield (sample, response) pairs.

        Args:
            samples: Iterator of JudgingSample objects to judge
            model_config: Model configuration with provider and settings

        Yields:
            Tuples of (JudgingSample, LLMResponse) for each inference

        Raises:
            ValueError: If configuration is invalid
            Exception: If provider API fails
        """
        pass
