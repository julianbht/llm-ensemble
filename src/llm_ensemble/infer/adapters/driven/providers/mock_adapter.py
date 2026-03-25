"""Mock LLM adapter for offline presentations and testing.

Returns hardcoded relevance judgements without making API calls.
Perfect for demos where internet access is not available or desired.
"""

from __future__ import annotations
import time
import random
from typing import Optional

from llm_ensemble.infer.domain.entities.llm_invocation_metrics import LLMInvocationMetrics
from llm_ensemble.infer.domain.entities.model_config import ModelConfig
from llm_ensemble.infer.domain.entities.retry_config_schema import RetryConfig
from llm_ensemble.infer.application.ports.driven.for_invoking_llm import ForInvokingLLM
from llm_ensemble.libs.logging.structlog_logger import get_logger


class MockLLMAdapter(ForInvokingLLM):
    """Mock LLM provider for offline presentations.

    Returns realistic-looking judgements without making API calls.
    Simulates latency and token usage for authenticity.
    """

    VERSION = "1.0"
    PROVIDER_NAME = "mock"

    # Realistic response templates for different relevance levels
    HIGHLY_RELEVANT_RESPONSES = [
        "Label: highly relevant\nConfidence: high\nRationale: The document directly addresses the query's main topic and provides comprehensive information that fully satisfies the information need.",
        "Label: highly relevant\nConfidence: high\nRationale: This document is exactly what the user is looking for, containing detailed and accurate information about the specific topic requested.",
        "Label: highly relevant\nConfidence: medium\nRationale: The document covers the query topic thoroughly, though some aspects could be more detailed. Overall, it provides strong relevance.",
    ]

    SOMEWHAT_RELEVANT_RESPONSES = [
        "Label: somewhat relevant\nConfidence: medium\nRationale: The document contains relevant information but also includes tangential content. It partially addresses the query but isn't fully focused on it.",
        "Label: somewhat relevant\nConfidence: high\nRationale: While the document discusses related topics, it doesn't fully answer the specific question posed in the query.",
        "Label: somewhat relevant\nConfidence: low\nRationale: The document has some connection to the query topic but the relevance is limited and indirect.",
    ]

    NOT_RELEVANT_RESPONSES = [
        "Label: not relevant\nConfidence: high\nRationale: The document discusses a completely different topic and does not address the query's information need at all.",
        "Label: not relevant\nConfidence: high\nRationale: Despite superficial keyword overlap, the document's content is unrelated to what the user is searching for.",
        "Label: not relevant\nConfidence: medium\nRationale: The document may mention related terms, but it does not provide meaningful information for this query.",
    ]

    def __init__(
        self,
        model_config: ModelConfig,
        retry_config: RetryConfig,
        base_latency_ms: float = 500.0,
        latency_variance_ms: float = 200.0,
        deterministic: bool = False,
    ):
        """Initialize mock LLM adapter.

        Args:
            model_config: Model configuration (used for metadata only)
            retry_config: Retry configuration (not used, but required by interface)
            base_latency_ms: Base simulated latency in milliseconds
            latency_variance_ms: Random variance added to latency
            deterministic: If True, always return the same response for consistency
        """
        self.model_config = model_config
        self.retry_config = retry_config
        self.base_latency_ms = base_latency_ms
        self.latency_variance_ms = latency_variance_ms
        self.deterministic = deterministic
        self.logger = get_logger(component=f"{self.PROVIDER_NAME}_provider")

        # Counter for deterministic mode
        self._call_count = 0

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
        """Return mock response with simulated latency and metrics.

        Args:
            prompt: Pre-built prompt string (analyzed for response selection)

        Returns:
            Tuple of (mock_response_text, invocation_metrics)
        """
        # Simulate API latency
        latency_ms = self._simulate_latency()

        # Select response based on prompt content or deterministically
        response_text = self._select_response(prompt)

        # Calculate realistic token counts
        prompt_tokens = self._estimate_tokens(prompt)
        completion_tokens = self._estimate_tokens(response_text)
        total_tokens = prompt_tokens + completion_tokens

        # Calculate mock costs if pricing is available
        cost_estimate_usd = None
        actual_cost_usd = None
        if self.model_config.pricing:
            prompt_cost = (prompt_tokens / 1_000_000) * self.model_config.pricing.prompt_cost_per_1m_tokens
            completion_cost = (completion_tokens / 1_000_000) * self.model_config.pricing.completion_cost_per_1m_tokens
            cost_estimate_usd = prompt_cost + completion_cost
            # Mock adapter has "perfect" cost accuracy
            actual_cost_usd = cost_estimate_usd

        # Create realistic metrics
        metrics = LLMInvocationMetrics(
            latency_ms=latency_ms,
            retries=0,  # Mock never retries
            cost_estimate_usd=cost_estimate_usd,
            actual_cost_usd=actual_cost_usd,
            generation_id=f"mock-gen-{self._call_count}",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

        self._call_count += 1
        return response_text, metrics

    def _simulate_latency(self) -> float:
        """Simulate realistic API latency.

        Returns:
            Simulated latency in milliseconds
        """
        if self.deterministic:
            latency_ms = self.base_latency_ms
        else:
            # Add random variance
            variance = random.uniform(-self.latency_variance_ms, self.latency_variance_ms)
            latency_ms = max(50.0, self.base_latency_ms + variance)

        # Actually sleep to simulate the latency
        time.sleep(latency_ms / 1000.0)
        return latency_ms

    def _select_response(self, prompt: str) -> str:
        """Select appropriate mock response based on prompt content.

        Analyzes the prompt to determine which relevance level to return.
        Falls back to cycling through responses in deterministic mode.

        Args:
            prompt: The prompt text to analyze

        Returns:
            Mock LLM response text
        """
        if self.deterministic:
            # Cycle through all responses deterministically
            all_responses = (
                self.HIGHLY_RELEVANT_RESPONSES +
                self.SOMEWHAT_RELEVANT_RESPONSES +
                self.NOT_RELEVANT_RESPONSES
            )
            return all_responses[self._call_count % len(all_responses)]

        # Try to infer relevance from prompt content
        # This is a simple heuristic - you can make it more sophisticated
        prompt_lower = prompt.lower()

        # Look for hints in the prompt
        if any(word in prompt_lower for word in ["highly", "perfect", "exact", "precisely"]):
            return random.choice(self.HIGHLY_RELEVANT_RESPONSES)
        elif any(word in prompt_lower for word in ["not", "irrelevant", "unrelated", "different"]):
            return random.choice(self.NOT_RELEVANT_RESPONSES)
        elif any(word in prompt_lower for word in ["somewhat", "partial", "related"]):
            return random.choice(self.SOMEWHAT_RELEVANT_RESPONSES)

        # Default: randomly distribute across relevance levels with realistic proportions
        # Typical IR datasets: ~30% highly relevant, ~40% somewhat, ~30% not relevant
        roll = random.random()
        if roll < 0.3:
            return random.choice(self.HIGHLY_RELEVANT_RESPONSES)
        elif roll < 0.7:
            return random.choice(self.SOMEWHAT_RELEVANT_RESPONSES)
        else:
            return random.choice(self.NOT_RELEVANT_RESPONSES)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Uses simple approximation: ~4 characters per token.
        This matches typical tokenization ratios for English text.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        return max(1, len(text) // 4)
