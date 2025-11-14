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

from llm_ensemble.infer.schemas.llm_judgement import LLMResponse
from llm_ensemble.infer.schemas import ModelConfig
from llm_ensemble.infer.schemas.retry_config_schema import RetryConfig
from llm_ensemble.infer.ports import LLMProvider


class OllamaAdapter(LLMProvider):
    """Ollama implementation of the LLMProvider port.

    Pure API client that sends pre-built prompts to Ollama and returns raw responses.
    Does NOT build prompts or parse responses - that's orchestrated by InferenceService.

    Example:
        >>> from llm_ensemble.infer.config_loaders import load_retry_config
        >>> retry_config = load_retry_config("standard")
        >>> adapter = OllamaAdapter(retry_config, base_url="http://localhost:11434")
        >>> response = adapter.infer("pre-built prompt", config)
        >>> print(response.raw_response)
    """

    def __init__(
        self,
        retry_config: RetryConfig,
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
        logger: Optional[structlog.stdlib.BoundLogger] = None,
    ):
        """Initialize Ollama adapter.

        Args:
            retry_config: Retry configuration for exponential backoff
            base_url: Ollama server URL (default: http://localhost:11434)
            timeout: Request timeout in seconds (default: 60)
            logger: Optional logger for retry events
        """
        super().__init__(retry_config, logger)

        self.base_url = base_url
        self.timeout = timeout

    def _do_infer(
        self,
        prompt: str,
        model_config: ModelConfig,
    ) -> LLMResponse:
        """Perform the actual Ollama API call (called by base class retry logic).

        Args:
            prompt: Pre-built prompt string (from PromptBuilder)
            model_config: Model configuration with provider and settings

        Returns:
            LLMResponse with raw response text and metadata

        Raises:
            NotImplementedError: Ollama adapter not yet implemented
            APIError: If API request fails (triggers retry in base class)
        """
        raise NotImplementedError("Ollama adapter not yet implemented")
