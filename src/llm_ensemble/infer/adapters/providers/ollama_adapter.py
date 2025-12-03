"""Ollama adapter for LLM inference.

Handles communication with local Ollama server and converts responses
to LLMResponse objects. Implements the LLMProvider port.

This is a PURE API client - it accepts pre-built prompts and returns raw responses.
It does NOT build prompts (that's PromptBuilder's job) or parse responses (that's
ResponseParser's job). The InferenceService orchestrates all port interactions.
"""

from __future__ import annotations
from typing import Optional
import structlog

from llm_ensemble.infer.entities.llm_invocation_metrics import LLMInvocationMetrics
from llm_ensemble.infer.schemas import ModelConfig
from llm_ensemble.infer.schemas.retry_config_schema import RetryConfig
from llm_ensemble.infer.entities.llm_invocation_dto import LLMInvocationDTO
from llm_ensemble.infer.ports import LLMProvider


class OllamaAdapter(LLMProvider):
    """Ollama implementation of the LLMProvider port.

    Pure API client that sends pre-built prompts to Ollama and returns raw responses.
    Does NOT build prompts or parse responses - that's orchestrated by InferenceService.
    """

    def __init__(
        self,
        provider_name: str,
        model_name: str,
        retry_config: RetryConfig,
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
        logger: Optional[structlog.stdlib.BoundLogger] = None,
    ):
        """Initialize Ollama adapter.

        Args:
            provider_name: Provider identifier (from config, e.g., 'ollama')
            model_name: Model identifier (from config, e.g., 'llama2')
            retry_config: Retry configuration for exponential backoff
            base_url: Ollama server URL (default: http://localhost:11434)
            timeout: Request timeout in seconds (default: 60)
            logger: Optional logger for retry events
        """
        super().__init__(provider_name, model_name, retry_config, logger)

        self.base_url = base_url
        self.timeout = timeout

    def _do_infer_raw(
        self,
        prompt: str,
        model_config: ModelConfig,
    ) -> LLMInvocationDTO:
        """Perform the actual Ollama API call (called by base class retry logic).

        Args:
            prompt: Pre-built prompt string (from PromptBuilder)
            model_config: Model configuration with provider and settings

        Returns:
            LLMInvocationDTO with response text and metrics (without retry count)

        Raises:
            NotImplementedError: Ollama adapter not yet implemented
            APIError: If API request fails (triggers retry in base class)
        """
        raise NotImplementedError("Ollama adapter not yet implemented")
