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

    Providers are PURE API clients - they accept pre-built prompts and return
    raw responses. They do NOT build prompts or parse responses. The domain
    service (InferenceService) orchestrates all port interactions.

    Providers yield (sample, LLMResponse) tuples - they do NOT attach manifests.
    The domain service handles manifest attachment to create final LLMJudgement objects.

    Example:
        >>> class OpenRouterAdapter(LLMProvider):
        ...     def __init__(self, api_key, ...):
        ...         self.api_key = api_key
        ...     def infer(self, sample_prompt_pairs, model_config):
        ...         for sample, prompt in sample_prompt_pairs:
        ...             response = self._call_api(prompt)  # Just send text to API
        ...             yield (sample, response)
    """

    @abstractmethod
    def infer(
        self,
        sample_prompt_pairs: Iterator[tuple[JudgingSample, str]],
        model_config: ModelConfig,
    ) -> Iterator[tuple[JudgingSample, LLMResponse]]:
        """Run inference on pre-built prompts and yield (sample, response) pairs.

        Args:
            sample_prompt_pairs: Iterator of (JudgingSample, prompt_string) tuples
                                where prompts have been pre-built by PromptBuilder
            model_config: Model configuration with provider and settings

        Yields:
            Tuples of (JudgingSample, LLMResponse) for each inference

        Raises:
            ValueError: If configuration is invalid
            Exception: If provider API fails
        """
        pass
