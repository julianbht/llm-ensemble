"""Ollama adapter for LLM inference."""

from __future__ import annotations

from llm_ensemble.infer.domain.entities.llm_invocation_metrics import LLMInvocationMetrics
from llm_ensemble.infer.domain.entities.model_config import ModelConfig
from llm_ensemble.infer.domain.entities.retry_config_schema import RetryConfig
from llm_ensemble.infer.application.ports.driven.for_invoking_llm import ForInvokingLLM
from llm_ensemble.libs.logging.structlog_logger import get_logger


class OllamaAdapter(ForInvokingLLM):
    """Ollama implementation of the LLMProvider port."""

    VERSION = "1.0"
    PROVIDER_NAME = "ollama"

    def __init__(
        self,
        model_config: ModelConfig,
        retry_config: RetryConfig,
        base_url: str = "http://localhost:11434",
    ):
        """Initialize Ollama adapter.

        Args:
            model_config: Complete model configuration
            retry_config: Retry configuration for exponential backoff and timeouts
            base_url: Ollama server URL (default: http://localhost:11434)
        """
        self.model_config = model_config
        self.retry_config = retry_config
        self.base_url = base_url
        self.logger = get_logger(component=f"{self.PROVIDER_NAME}_provider")

    def get_provider(self):
        """Get Provider metadata for this adapter.

        Returns:
            Provider entity with name and version
        """
        from llm_ensemble.infer.domain.entities.provider import Provider
        return Provider(name=self.PROVIDER_NAME, version=self.VERSION)

    def get_model_config(self) -> ModelConfig:
        """Get model configuration for this provider.

        Returns:
            ModelConfig entity used for inference
        """
        return self.model_config

    def get_retry_config(self) -> RetryConfig:
        """Get retry configuration for this provider.

        Returns:
            RetryConfig entity used for exponential backoff
        """
        return self.retry_config

    def infer(
        self,
        prompt: str,
    ) -> tuple[str, LLMInvocationMetrics]:
        """Perform Ollama API call.

        Args:
            prompt: Pre-built prompt string

        Returns:
            Tuple of (raw_response_text, invocation_metrics)

        Raises:
            NotImplementedError: Ollama adapter not yet implemented
        """
        raise NotImplementedError("Ollama adapter not yet implemented")
