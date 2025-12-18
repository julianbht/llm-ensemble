"""Retry wrapper for LLM providers.

Implements exponential backoff with jitter as a decorator around provider adapters.
This separates retry concerns from provider implementation (composition over inheritance).
"""

from __future__ import annotations
import random
import time
from openai import APIError

from llm_ensemble.infer.application.ports.driven.llm_provider_port import LLMProviderPort
from llm_ensemble.infer.schemas.retry_config_schema import RetryConfig
from llm_ensemble.infer.domain.entities.llm_judgement import LLMInvocationMetrics
from llm_ensemble.libs.logging import get_logger
from llm_ensemble.libs.logging.log_events import InferLogEvent


class RetryingProvider(LLMProviderPort):
    """Wrapper that adds retry logic to any LLM provider.

    Implements exponential backoff with jitter for retryable errors.
    Delegates actual inference to the wrapped provider.

    This uses composition (wrapping) to separate retry concerns from provider
    implementation, while still implementing the LLMProviderPort interface.
    """

    def __init__(self, provider: LLMProviderPort, retry_config: RetryConfig):
        """Initialize wrapper with provider and retry configuration.

        Args:
            provider: The actual LLM provider adapter to wrap
            retry_config: Retry configuration for exponential backoff
        """
        self.provider = provider
        self.retry_config = retry_config
        self.logger = get_logger(component=f"{provider.provider_name}_retry")

    def get_provider(self):
        """Delegate to wrapped provider."""
        return self.provider.get_provider()

    @property
    def provider_name(self) -> str:
        """Delegate to wrapped provider."""
        return self.provider.provider_name

    @property
    def model_name(self) -> str:
        """Delegate to wrapped provider."""
        return self.provider.model_name

    def infer(
        self,
        prompt: str,
    ) -> tuple[str, LLMInvocationMetrics]:
        """Run inference with automatic retry logic.

        Implements exponential backoff with jitter. Delegates actual inference
        to wrapped provider and adds retry count to metrics.

        Args:
            prompt: Pre-built prompt string (from PromptBuilder)

        Returns:
            Tuple of (raw_response_text, invocation_metrics) with retry count set

        Raises:
            ValueError: If configuration is invalid
            Exception: If provider API fails after all retries exhausted
        """
        # Retry loop with exponential backoff
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                # Call wrapped provider (model_config was passed at initialization)
                raw_response_text, metrics = self.provider.infer(prompt)

                # Add retry count to metrics (create new instance with retry count)
                metrics_with_retries = LLMInvocationMetrics(
                    latency_ms=metrics.latency_ms,
                    retries=attempt,
                    cost_estimate_usd=metrics.cost_estimate_usd,
                    generation_id=metrics.generation_id,
                    prompt_tokens=metrics.prompt_tokens,
                    completion_tokens=metrics.completion_tokens,
                    total_tokens=metrics.total_tokens,
                )

                return raw_response_text, metrics_with_retries

            except APIError as e:
                # Check if we should retry
                is_retryable = False
                if hasattr(e, 'status_code'):
                    is_retryable = e.status_code in self.retry_config.retryable_status_codes

                # If not retryable or out of retries, raise
                if not is_retryable or attempt >= self.retry_config.max_retries:
                    # Log the final failure
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
                jitter = random.uniform(0, delay * 0.1)  # 10% jitter
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

                # Sleep before retry
                time.sleep(total_delay)

        # Should never reach here, but for type safety
        raise RuntimeError("Retry loop exited unexpectedly")
