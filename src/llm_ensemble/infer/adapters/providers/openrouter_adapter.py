"""OpenRouter adapter for LLM inference.

Handles HTTP communication with OpenRouter API and converts responses
to LLMResponse objects. Implements the LLMProvider port.

This is a PURE API client - it accepts pre-built prompts and returns raw responses.
It does NOT build prompts (that's PromptBuilder's job) or parse responses (that's
ResponseParser's job). The InferenceService orchestrates all port interactions.
"""

from __future__ import annotations
import os
import random
import time
from typing import Optional
from openai import OpenAI, APIError

from llm_ensemble.infer.schemas.llm_judgement import LLMResponse
from llm_ensemble.infer.schemas import ModelConfig
from llm_ensemble.infer.schemas.warnings import ProviderWarning, ProviderWarningCode
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
        api_key: Optional[str] = None,
        timeout: int = 30,
    ):
        """Initialize OpenRouter adapter.

        Args:
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            timeout: Request timeout in seconds (default: 30)
        """
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
    ) -> LLMResponse:
        """Run inference on a single pre-built prompt using OpenRouter API.

        Args:
            prompt: Pre-built prompt string (from PromptBuilder)
            model_config: Model configuration with provider and settings

        Returns:
            LLMResponse with raw response text and metadata

        Raises:
            ValueError: If openrouter_model_id is not configured
            Exception: If API request fails
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

        # Track timing and retries
        warnings: list = []  # Will be populated with ProviderWarning objects if needed
        start_time = time.time()
        retry_count = 0

        # Hardcoded retry configuration
        max_retries = 5
        base_delay = 1.0  # seconds
        max_delay = 60.0  # seconds

        # Retry loop with exponential backoff
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                # Send request with all configured parameters
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    **api_params
                )

                # Success! Break out of retry loop
                break

            except APIError as e:
                last_error = e
                retry_count = attempt

                # Check if we should retry
                is_retryable = False
                if hasattr(e, 'status_code'):
                    # Retry on rate limits (429) and server errors (503, 504)
                    is_retryable = e.status_code in [429, 503, 504]

                # If not retryable or out of retries, raise
                if not is_retryable or attempt >= max_retries:
                    warnings.append(
                        ProviderWarning(
                            code=ProviderWarningCode.RETRY_FAILED,
                            message=f"Request failed after {retry_count} retries: {str(e)}",
                            metadata={
                                "retry_count": retry_count,
                                "error_type": type(e).__name__,
                                "status_code": getattr(e, 'status_code', None),
                            }
                        )
                    )
                    raise

                # Calculate exponential backoff with jitter
                delay = min(base_delay * (2 ** attempt), max_delay)
                jitter = random.uniform(0, delay * 0.1)  # 10% jitter
                total_delay = delay + jitter

                # Log retry warning
                warnings.append(
                    ProviderWarning(
                        code=ProviderWarningCode.API_ERROR,
                        message=f"Rate limited (attempt {attempt + 1}/{max_retries + 1}), retrying after {total_delay:.2f}s",
                        metadata={
                            "attempt": attempt + 1,
                            "backoff_seconds": total_delay,
                            "error_type": type(e).__name__,
                            "status_code": getattr(e, 'status_code', None),
                        }
                    )
                )

                # Sleep before retry
                time.sleep(total_delay)

        latency_ms = (time.time() - start_time) * 1000

        # Extract response text
        raw_response = response.choices[0].message.content

        # Build LLMResponse with RAW output only (no parsing)
        # The InferenceService will handle parsing via ResponseParser
        llm_response = LLMResponse(
            raw_response=raw_response,
            latency_ms=latency_ms,
            cost_estimate_usd=None,  # Could be added later
            warnings=warnings,
            retries=retry_count,  # Track retry count
        )

        return llm_response
