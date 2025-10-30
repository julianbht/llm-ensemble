"""Port interface for LLM inference providers.

Defines the abstract contract that all LLM provider adapters must implement.
This allows the orchestrator to depend on an abstraction rather than concrete
provider implementations (OpenRouter, Ollama, HuggingFace, etc.).
"""

from __future__ import annotations
from abc import ABC, abstractmethod

from llm_ensemble.infer.schemas.llm_response import LLMResponse
from llm_ensemble.infer.schemas import ModelConfig


class LLMProvider(ABC):
    """Abstract base class for LLM inference providers.

    All provider adapters (OpenRouter, Ollama, HuggingFace) must inherit
    from this class and implement the infer() method.

    Providers are PURE API clients - they accept pre-built prompts and return
    raw responses. They do NOT build prompts or parse responses. The domain
    service (InferenceService) orchestrates all port interactions.

    Providers handle single inference requests. If a provider's API supports
    batching, the adapter can implement internal buffering transparently.

    Example:
        >>> class OpenRouterAdapter(LLMProvider):
        ...     def __init__(self, api_key, ...):
        ...         self.api_key = api_key
        ...     def infer(self, prompt, model_config):
        ...         response = self._call_api(prompt)  # Just send text to API
        ...         return response
    """

    @abstractmethod
    def infer(
        self,
        prompt: str,
        model_config: ModelConfig,
    ) -> LLMResponse:
        """Run inference on a single pre-built prompt.

        Args:
            prompt: Pre-built prompt string (from PromptBuilder)
            model_config: Model configuration with provider and settings

        Returns:
            LLMResponse with raw response text and metadata

        Raises:
            ValueError: If configuration is invalid
            Exception: If provider API fails
        """
        pass
