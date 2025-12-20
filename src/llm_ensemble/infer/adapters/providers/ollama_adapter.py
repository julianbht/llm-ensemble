"""Ollama adapter for LLM inference."""

from __future__ import annotations

from llm_ensemble.infer.domain.entities.llm_invocation_metrics import LLMInvocationMetrics
from llm_ensemble.infer.domain.entities.model_config import ModelConfig
from llm_ensemble.infer.schemas.retry_config_schema import RetryConfig
from llm_ensemble.infer.application.ports.driven.llm_provider_port import LLMProviderPort


class OllamaAdapter(LLMProviderPort):
    """Ollama implementation of the LLMProvider port."""

    VERSION = "1.0"

    def __init__(
        self,
        provider_name: str,
        model_config: ModelConfig,
        retry_config: RetryConfig,
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
    ):
        """Initialize Ollama adapter.

        Args:
            provider_name: Provider identifier (from config, e.g., 'ollama')
            model_config: Complete model configuration
            retry_config: Retry configuration for exponential backoff
            base_url: Ollama server URL (default: http://localhost:11434)
            timeout: Request timeout in seconds (default: 60)
        """
        super().__init__(provider_name, model_config, retry_config)

        self.base_url = base_url
        self.timeout = timeout

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
