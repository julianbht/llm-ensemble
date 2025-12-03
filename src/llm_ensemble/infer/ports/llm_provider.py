"""Port interface for LLM inference providers.

Defines the abstract contract that all LLM provider adapters must implement.
This allows the orchestrator to depend on an abstraction rather than concrete
provider implementations (OpenRouter, Ollama, HuggingFace, etc.).

Template Method Pattern:
- infer() is a concrete method implementing retry logic (template method)
- _do_infer() is abstract - providers implement the actual API call
- Retry logic is centralized in the base class, applied to all providers
"""

from __future__ import annotations
import random
import time
from abc import ABC, abstractmethod
from typing import Optional
from openai import APIError
import structlog

from llm_ensemble.infer.entities.llm_judgement import LLMInvocationMetrics
from llm_ensemble.infer.schemas import ModelConfig
from llm_ensemble.infer.schemas.retry_config_schema import RetryConfig
from llm_ensemble.infer.schemas.llm_invocation_dto import LLMInvocationDTO
from llm_ensemble.libs.logging.log_events import InferLogEvent


class LLMProvider(ABC):
    """Abstract base class for LLM inference providers with retry logic.

    Template Method Pattern:
    - infer() is a CONCRETE method with retry logic (implemented in base class)
    - _do_infer() is ABSTRACT - subclasses implement the actual API call
    - All providers automatically get exponential backoff with jitter

    Providers are PURE API clients - they accept pre-built prompts and return
    raw responses. They do NOT build prompts or parse responses. The domain
    service (InferenceService) orchestrates all port interactions.

    Providers handle single inference requests. If a provider's API supports
    batching, the adapter can implement internal buffering transparently.
    """

    def __init__(
        self,
        provider_name: str,
        model_name: str,
        retry_config: RetryConfig,
        logger: Optional[structlog.stdlib.BoundLogger] = None,
    ):
        """Initialize provider with identity from config.

        Args:
            provider_name: Provider identifier (from config, e.g., 'openrouter')
            model_name: Model identifier (from config, e.g., 'llama-4-maverick:free')
            retry_config: Retry configuration for exponential backoff
            logger: Optional logger for retry events (if None, no logging)
        """
        self.provider_name = provider_name
        self.model_name = model_name
        self.retry_config = retry_config
        self.logger = logger

    def infer(
        self,
        prompt: str,
        model_config: ModelConfig,
    ) -> tuple[str, LLMInvocationMetrics]:
        """Run inference with automatic retry logic (template method).

        This concrete method implements exponential backoff with jitter.
        It calls the abstract _do_infer_raw() method for the actual API call
        and maps the DTO to domain objects.

        Args:
            prompt: Pre-built prompt string (from PromptBuilder)
            model_config: Model configuration with provider and settings

        Returns:
            Tuple of (raw_response_text, invocation_metrics) with retry count set

        Raises:
            ValueError: If configuration is invalid
            Exception: If provider API fails after all retries exhausted
        """

        # Retry loop with exponential backoff
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                # Call the provider-specific implementation (returns DTO)
                invocation_dto = self._do_infer_raw(prompt, model_config)
                
                # Map DTO to domain objects
                raw_response_text = invocation_dto.response_text
                metrics = LLMInvocationMetrics(
                    latency_ms=invocation_dto.latency_ms,
                    retries=attempt,
                    cost_estimate_usd=invocation_dto.cost_estimate_usd,
                    generation_id=invocation_dto.generation_id,
                    prompt_tokens=invocation_dto.prompt_tokens,
                    completion_tokens=invocation_dto.completion_tokens,
                    total_tokens=invocation_dto.total_tokens,
                )

                return raw_response_text, metrics

            except APIError as e:

                # Check if we should retry
                is_retryable = False
                if hasattr(e, 'status_code'):
                    is_retryable = e.status_code in self.retry_config.retryable_status_codes

                # If not retryable or out of retries, raise
                if not is_retryable or attempt >= self.retry_config.max_retries:
                    # Log the final failure if logger available
                    if self.logger:
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

                # Log retry attempt if logger available
                if self.logger:
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

    @abstractmethod
    def _do_infer_raw(
        self,
        prompt: str,
        model_config: ModelConfig,
    ) -> LLMInvocationDTO:
        """Perform the actual inference API call (implemented by subclasses).

        This is the method that provider adapters must implement.
        It should make the API call and return a DTO with pure adapter output.

        The base class infer() method handles retries and maps DTO to domain objects.

        Args:
            prompt: Pre-built prompt string (from PromptBuilder)
            model_config: Model configuration with provider and settings

        Returns:
            LLMInvocationDTO with response text and metrics (without retry count)

        Raises:
            APIError: If API call fails (will trigger retry in base class)
            ValueError: If configuration is invalid
        """
        pass
