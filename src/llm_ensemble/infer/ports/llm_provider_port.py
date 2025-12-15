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

    Providers are PURE API clients - they accept pre-built prompts and return
    raw responses. They do NOT build prompts or parse responses. The domain
    service (InferenceService) orchestrates all port interactions.

    Retry logic is handled by a separate RetryingProvider wrapper to keep
    this interface clean and focused.

    Providers handle single inference requests. If a provider's API supports
    batching, the adapter can implement internal buffering transparently.
    """

    def __init__(
        self,
        provider_name: str,
        model_config: ModelConfig,
    ):
        """Initialize provider with configuration.

        Args:
            provider_name: Provider identifier (from config, e.g., 'openrouter')
            model_config: Complete model configuration (model_id, temperature, max_tokens, etc.)
        """
        self.provider_name = provider_name
        self.model_config = model_config
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

        Adapters implement the actual API call using the model_config provided at initialization.
        Returns the response text and invocation metrics (without retry count - that's handled by wrapper).

        Args:
            prompt: Pre-built prompt string (from PromptBuilder)

        Returns:
            Tuple of (raw_response_text, invocation_metrics)

        Raises:
            APIError: If API call fails
            ValueError: If configuration is invalid
        """
        pass
