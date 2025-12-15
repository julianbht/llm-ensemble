"""Ollama adapter for LLM inference.

Handles communication with local Ollama server and converts responses
to LLMResponse objects. Implements the LLMProvider port.

This is a PURE API client - it accepts pre-built prompts and returns raw responses.
It does NOT build prompts (that's PromptBuilder's job) or parse responses (that's
ResponseParser's job). The InferenceService orchestrates all port interactions.
"""

from __future__ import annotations
from typing import Optional

from llm_ensemble.infer.schemas.entities.llm_invocation_metrics import LLMInvocationMetrics
from llm_ensemble.infer.schemas.model_config_schema import ModelConfig
from llm_ensemble.infer.ports.llm_provider_port import LLMProviderPort


class OllamaAdapter(LLMProviderPort):
    """Ollama implementation of the LLMProvider port.

    Pure API client that sends pre-built prompts to Ollama and returns raw responses.
    Does NOT build prompts or parse responses - that's orchestrated by InferenceService.
    """

    def __init__(
        self,
        provider_name: str,
        model_name: str,
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
    ):
        """Initialize Ollama adapter.

        Args:
            provider_name: Provider identifier (from config, e.g., 'ollama')
            model_name: Model identifier (from config, e.g., 'llama2')
            base_url: Ollama server URL (default: http://localhost:11434)
            timeout: Request timeout in seconds (default: 60)
        """
        super().__init__(provider_name, model_name)

        self.base_url = base_url
        self.timeout = timeout

    def infer(
        self,
        prompt: str,
        model_config: ModelConfig,
    ) -> tuple[str, LLMInvocationMetrics]:
        """Perform Ollama API call and return response.

        Args:
            prompt: Pre-built prompt string (from PromptBuilder)
            model_config: Model configuration with provider and settings

        Returns:
            Tuple of (raw_response_text, invocation_metrics)

        Raises:
            NotImplementedError: Ollama adapter not yet implemented
            APIError: If API request fails
        """
        raise NotImplementedError("Ollama adapter not yet implemented")
