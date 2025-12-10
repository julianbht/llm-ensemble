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

from llm_ensemble.infer.schemas.entities.llm_invocation_metrics import LLMInvocationMetrics
from llm_ensemble.infer.schemas import ModelConfig
from llm_ensemble.infer.ports import LLMProviderPort


class OpenRouterAdapter(LLMProviderPort):
    """OpenRouter implementation of the LLMProvider port.

    Pure API client that sends pre-built prompts to OpenRouter and returns raw responses.
    Does NOT build prompts or parse responses - that's orchestrated by InferenceService.
    """

    def __init__(
        self,
        provider_name: str,
        model_name: str,
        api_key: Optional[str] = None,
        timeout: int = 30,
    ):
        """Initialize OpenRouter adapter with identity from config.

        Args:
            provider_name: Provider identifier (from config, e.g., 'openrouter')
            model_name: Model identifier (from config, e.g., 'llama-4-maverick:free')
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            timeout: Request timeout in seconds (default: 30)
        """
        super().__init__(provider_name, model_name)

        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.timeout = timeout

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY env var "
                "or pass api_key parameter."
            )

    def infer(
        self,
        prompt: str,
        model_config: ModelConfig,
    ) -> tuple[str, LLMInvocationMetrics]:
        """Perform OpenRouter API call and return response.

        Args:
            prompt: Pre-built prompt string (from PromptBuilder)
            model_config: Model configuration with inference parameters

        Returns:
            Tuple of (raw_response_text, invocation_metrics)

        Raises:
            APIError: If API request fails
        """
        # Build API parameters from config
        api_params = {
            "model": model_config.model_id,
        }

        # Add core inference parameters if set
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

        # Add additional parameters (stop sequences, response_format, etc.)
        if model_config.additional_params:
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
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            **api_params
        )

        latency_ms = (time.time() - start_time) * 1000

        # Extract response text
        raw_response_text = response.choices[0].message.content

        # Extract generation ID for async cost queries
        generation_id = getattr(response, "id", None)

        # Extract token usage
        prompt_tokens = None
        completion_tokens = None
        total_tokens = None
        if hasattr(response, "usage") and response.usage:
            prompt_tokens = getattr(response.usage, "prompt_tokens", None)
            completion_tokens = getattr(response.usage, "completion_tokens", None)
            total_tokens = getattr(response.usage, "total_tokens", None)

        # Calculate cost estimate from token usage and pricing config
        cost_estimate_usd = None
        if model_config.pricing and prompt_tokens is not None and completion_tokens is not None:
            prompt_cost = (prompt_tokens / 1_000_000) * model_config.pricing.prompt_cost_per_1m_tokens
            completion_cost = (completion_tokens / 1_000_000) * model_config.pricing.completion_cost_per_1m_tokens
            cost_estimate_usd = prompt_cost + completion_cost

        # Create metrics (retry count will be added by wrapper)
        metrics = LLMInvocationMetrics(
            latency_ms=latency_ms,
            retries=0,  # Will be set by RetryingProvider wrapper
            cost_estimate_usd=cost_estimate_usd,
            generation_id=generation_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

        return raw_response_text, metrics
