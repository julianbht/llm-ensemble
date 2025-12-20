"""Port interface for LLM inference providers.

Defines the abstract contract that all LLM provider adapters must implement.
This allows the orchestrator to depend on an abstraction rather than concrete
provider implementations (OpenRouter, Ollama, HuggingFace, etc.).

Pure interface - retry logic is handled by a separate wrapper.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from llm_ensemble.infer.domain.entities.llm_judgement import LLMInvocationMetrics
from llm_ensemble.infer.domain.entities.model_config import ModelConfig
from llm_ensemble.libs.logging import get_logger

if TYPE_CHECKING:
    from llm_ensemble.infer.domain.entities.provider import Provider


class LLMProviderPort(ABC):
    """Abstract interface for LLM inference providers.

    Providers accept pre-built prompts and return raw responses.
    They do NOT build prompts or parse responses. The application
    use case (InferenceApplication) orchestrates all port interactions.
    """

    def __init__(
        self,
        provider_name: str,
        model_config: ModelConfig,
        retry_config: RetryConfig,
    ):
        """Initialize provider with configuration.

        Args:
            provider_name: Provider identifier (from config, e.g., 'openrouter')
            model_config: Complete model configuration (model_id, temperature, max_tokens, etc.)
            retry_config: Retry configuration for exponential backoff
        """
        self.provider_name = provider_name
        self.model_config = model_config
        self.retry_config = retry_config
        self.logger = get_logger(component=f"{provider_name}_provider")

    def get_provider(self) -> Provider:
        """Get Provider domain object for this provider.

        Returns:
            Provider entity with random UUID and provider name
        """
        from llm_ensemble.infer.schemas.entities.provider import Provider
        return Provider(name=self.provider_name)

    @abstractmethod
    def infer(
        self,
        prompt: str,
    ) -> tuple[str, LLMInvocationMetrics]:
        """Perform inference and return response with metrics.

        Adapters implement the actual API call using the model_config and retry_config.
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
