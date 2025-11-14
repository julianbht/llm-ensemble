"""OpenRouter adapter for LLM inference.

Handles HTTP communication with OpenRouter API and converts responses
to LLMResponse objects. Implements the LLMProvider port.

This is a PURE API client - it accepts pre-built prompts and returns raw responses.
It does NOT build prompts (that's PromptBuilder's job) or parse responses (that's
ResponseParser's job). The InferenceService orchestrates all port interactions.
"""

from __future__ import annotations
import os
import time
from typing import Optional
from openai import OpenAI
import structlog

from llm_ensemble.infer.schemas.llm_judgement import LLMResponse
from llm_ensemble.infer.schemas import ModelConfig
from llm_ensemble.infer.schemas.retry_config_schema import RetryConfig
from llm_ensemble.infer.ports import LLMProvider


class OpenRouterAdapter(LLMProvider):
    """OpenRouter implementation of the LLMProvider port.

    Pure API client that sends pre-built prompts to OpenRouter and returns raw responses.
    Does NOT build prompts or parse responses - that's orchestrated by InferenceService.

    Example:
        >>> from llm_ensemble.infer.config_loaders import load_model_config
        >>> config = load_model_config("gpt-oss-20b")
        >>> adapter = OpenRouterAdapter(api_key="...")
        >>> response = adapter.infer("pre-built prompt", config)
        >>> print(response.raw_response)  # Unparsed text
    """

    def __init__(
        self,
        retry_config: RetryConfig,
        api_key: Optional[str] = None,
        timeout: int = 30,
        logger: Optional[structlog.stdlib.BoundLogger] = None,
    ):
        """Initialize OpenRouter adapter.

        Args:
            retry_config: Retry configuration for exponential backoff
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            timeout: Request timeout in seconds (default: 30)
            logger: Optional logger for retry events
        """
        super().__init__(retry_config, logger)

        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.timeout = timeout

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY env var "
                "or pass api_key parameter."
            )

    def _do_infer(
        self,
        prompt: str,
        model_config: ModelConfig,
    ) -> LLMResponse:
        """Perform the actual OpenRouter API call (called by base class retry logic).

        Args:
            prompt: Pre-built prompt string (from PromptBuilder)
            model_config: Model configuration with provider and settings

        Returns:
            LLMResponse with raw response text and metadata

        Raises:
            ValueError: If openrouter_model_id is not configured
            APIError: If API request fails (triggers retry in base class)
        """
        if not model_config.openrouter_model_id:
            raise ValueError(
                f"Model {model_config.model_id} is configured for OpenRouter "
                f"but missing openrouter_model_id field"
            )

        # Build API parameters from explicit config fields
        api_params = {
            "model": model_config.openrouter_model_id,
        }

        # Add explicit parameters if set
        if model_config.temperature is not None:
            api_params["temperature"] = model_config.temperature
        if model_config.max_tokens is not None:
            api_params["max_tokens"] = model_config.max_tokens
        if model_config.top_p is not None:
            api_params["top_p"] = model_config.top_p
        if model_config.frequency_penalty is not None:
            api_params["frequency_penalty"] = model_config.frequency_penalty
        if model_config.presence_penalty is not None:
            api_params["presence_penalty"] = model_config.presence_penalty
        if model_config.seed is not None:
            api_params["seed"] = model_config.seed
        if model_config.stop is not None:
            api_params["stop"] = model_config.stop
        if model_config.response_format is not None:
            api_params["response_format"] = model_config.response_format

        # Add additional parameters (advanced/provider-specific)
        api_params.update(model_config.additional_params)

        # Initialize OpenAI client configured for OpenRouter
        client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=self.timeout,
        )

        # Track timing
        start_time = time.time()

        # Send request with all configured parameters
        # Note: APIError exceptions will be caught and handled by base class retry logic
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            **api_params
        )

        latency_ms = (time.time() - start_time) * 1000

        # Extract response text
        raw_response = response.choices[0].message.content

        # Note: retry count will be added by base class
        llm_response = LLMResponse(
            raw_response=raw_response,
            latency_ms=latency_ms,
            cost_estimate_usd=None,  # Could be added later
        )

        return llm_response
