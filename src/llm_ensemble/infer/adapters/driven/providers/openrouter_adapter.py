"""OpenRouter adapter for LLM inference.

Handles HTTP communication with OpenRouter API and converts responses
to LLMResponse objects. Implements the LLMProvider port.
"""

from __future__ import annotations
import os
import time
import random
from typing import Optional
from openai import OpenAI, APIError

from llm_ensemble.infer.domain.entities.llm_invocation_metrics import LLMInvocationMetrics
from llm_ensemble.infer.domain.entities.model_config import ModelConfig
from llm_ensemble.infer.domain.entities.retry_config_schema import RetryConfig
from llm_ensemble.infer.application.ports.driven.llm_provider_port import LLMProviderPort
from llm_ensemble.libs.logging import get_logger
from llm_ensemble.libs.logging.log_events import InferLogEvent


class OpenRouterAdapter(LLMProviderPort):
    """OpenRouter implementation of the LLMProvider port."""

    VERSION = "1.0"
    PROVIDER_NAME = "openrouter"

    def __init__(
        self,
        model_config: ModelConfig,
        retry_config: RetryConfig,
        api_key: Optional[str] = None,
        timeout: int = 30,
    ):
        """Initialize OpenRouter adapter.

        Args:
            model_config: Complete model configuration
            retry_config: Retry configuration for exponential backoff
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            timeout: Request timeout in seconds (default: 30)
        """
        self.model_config = model_config
        self.retry_config = retry_config
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.timeout = timeout
        self.logger = get_logger(component=f"{self.PROVIDER_NAME}_provider")

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY env var "
                "or pass api_key parameter."
            )

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
        """Perform OpenRouter API call with retry logic.

        Args:
            prompt: Pre-built prompt string

        Returns:
            Tuple of (raw_response_text, invocation_metrics)

        Raises:
            APIError: If API request fails after all retries
        """
        # Retry loop with exponential backoff
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                return self._make_api_call(prompt, attempt)
            except APIError as e:
                # Check if we should retry
                is_retryable = False
                if hasattr(e, 'status_code'):
                    is_retryable = e.status_code in self.retry_config.retryable_status_codes

                # If not retryable or out of retries, raise
                if not is_retryable or attempt >= self.retry_config.max_retries:
                    self.logger.warning(
                        InferLogEvent.RETRY_EXHAUSTED,
                        retry_count=attempt,
                        error_type=type(e).__name__,
                        status_code=getattr(e, 'status_code', None),
                    )
                    raise

                # Calculate exponential backoff with jitter
                delay = min(
                    self.retry_config.base_delay_seconds * (2 ** attempt),
                    self.retry_config.max_delay_seconds
                )
                jitter = random.uniform(0, delay * 0.1)
                total_delay = delay + jitter

                # Log retry attempt
                self.logger.info(
                    InferLogEvent.RETRY_ATTEMPT,
                    attempt=attempt + 1,
                    max_retries=self.retry_config.max_retries + 1,
                    backoff_seconds=round(total_delay, 2),
                    error_type=type(e).__name__,
                    status_code=getattr(e, 'status_code', None),
                )

                time.sleep(total_delay)

        raise RuntimeError("Retry loop exited unexpectedly")

    def _make_api_call(
        self,
        prompt: str,
        retry_attempt: int,
    ) -> tuple[str, LLMInvocationMetrics]:
        """Make actual API call to OpenRouter.

        Args:
            prompt: Pre-built prompt string
            retry_attempt: Current retry attempt number

        Returns:
            Tuple of (raw_response_text, invocation_metrics)
        """
        # Build API parameters from config
        api_params = {
            "model": self.model_config.model_id,
        }

        # Add core inference parameters if set
        if self.model_config.temperature is not None:
            api_params["temperature"] = self.model_config.temperature
        if self.model_config.max_tokens is not None:
            api_params["max_tokens"] = self.model_config.max_tokens
        if self.model_config.top_p is not None:
            api_params["top_p"] = self.model_config.top_p
        if self.model_config.frequency_penalty is not None:
            api_params["frequency_penalty"] = self.model_config.frequency_penalty
        if self.model_config.presence_penalty is not None:
            api_params["presence_penalty"] = self.model_config.presence_penalty
        if self.model_config.seed is not None:
            api_params["seed"] = self.model_config.seed

        # Add additional parameters
        if self.model_config.additional_params:
            api_params.update(self.model_config.additional_params)

        # Initialize OpenAI client configured for OpenRouter
        client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=self.timeout,
        )

        # Track timing
        start_time = time.time()

        # Send request
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            **api_params
        )

        latency_ms = (time.time() - start_time) * 1000

        # Extract response text
        raw_response_text = response.choices[0].message.content

        # Extract metadata
        generation_id = getattr(response, "id", None)
        prompt_tokens = None
        completion_tokens = None
        total_tokens = None
        if hasattr(response, "usage") and response.usage:
            prompt_tokens = getattr(response.usage, "prompt_tokens", None)
            completion_tokens = getattr(response.usage, "completion_tokens", None)
            total_tokens = getattr(response.usage, "total_tokens", None)

        # Calculate cost estimate
        cost_estimate_usd = None
        if self.model_config.pricing and prompt_tokens is not None and completion_tokens is not None:
            prompt_cost = (prompt_tokens / 1_000_000) * self.model_config.pricing.prompt_cost_per_1m_tokens
            completion_cost = (completion_tokens / 1_000_000) * self.model_config.pricing.completion_cost_per_1m_tokens
            cost_estimate_usd = prompt_cost + completion_cost
            
            # Log cost at the source where it's calculated
            self.logger.info(
                InferLogEvent.COST_CALCULATED,
                cost_estimate_usd=cost_estimate_usd,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        # Create metrics
        metrics = LLMInvocationMetrics(
            latency_ms=latency_ms,
            retries=retry_attempt,
            cost_estimate_usd=cost_estimate_usd,
            generation_id=generation_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

        return raw_response_text, metrics
