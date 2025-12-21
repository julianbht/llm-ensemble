"""Port interface for LLM inference providers.

Defines the abstract contract that all LLM provider adapters must implement.
This allows the orchestrator to depend on an abstraction rather than concrete
provider implementations (OpenRouter, Ollama, HuggingFace, etc.).

Pure interface - no constructor, only abstract methods.
Each adapter defines its own constructor with whatever it needs.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

from llm_ensemble.infer.domain.entities.llm_judgement import LLMInvocationMetrics
from llm_ensemble.infer.domain.entities.model_config import ModelConfig
from llm_ensemble.infer.domain.entities.provider import Provider
from llm_ensemble.infer.domain.entities.retry_config_schema import RetryConfig


class LLMProviderPort(ABC):
    """Abstract interface for LLM inference providers.

    Pure interface - adapters define their own constructors.

    Providers accept pre-built prompts and return raw responses.
    They do NOT build prompts or parse responses. The application
    use case (InferenceApplication) orchestrates all port interactions.
    """

    @abstractmethod
    def get_provider(self) -> Provider:
        """Get Provider metadata for this adapter.

        Adapter knows its own identity and version.

        Returns:
            Provider entity with name and version
        """
        pass

    @abstractmethod
    def get_model_config(self) -> ModelConfig:
        """Get model configuration for this provider.

        Returns:
            ModelConfig entity used for inference
        """
        pass

    @abstractmethod
    def get_retry_config(self) -> RetryConfig:
        """Get retry configuration for this provider.

        Returns:
            RetryConfig entity used for exponential backoff
        """
        pass

    @abstractmethod
    def infer(
        self,
        prompt: str,
    ) -> tuple[str, LLMInvocationMetrics]:
        """Perform inference and return response with metrics.

        Adapters implement the actual API call using their model_config and retry_config.
        Returns the response text and invocation metrics (including retry count).

        Args:
            prompt: Pre-built prompt string (from PromptBuilder)

        Returns:
            Tuple of (raw_response_text, invocation_metrics)

        Raises:
            APIError: If API call fails after all retries
            ValueError: If configuration is invalid
        """
        pass
