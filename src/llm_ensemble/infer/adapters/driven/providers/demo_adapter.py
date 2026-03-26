"""Demo LLM adapter for offline presentations and testing.

Returns real LLM responses from a pre-extracted lookup file, matched by
prompt content hash. No API calls needed.

The lookup file contains prompt_hash → response mappings extracted from
actual experiment runs.
"""

from __future__ import annotations
import hashlib
import json
import time
import random
from pathlib import Path

from llm_ensemble.infer.domain.entities.llm_invocation_metrics import (
    LLMInvocationMetrics,
)
from llm_ensemble.infer.domain.entities.model_config import ModelConfig
from llm_ensemble.infer.domain.entities.retry_config_schema import RetryConfig
from llm_ensemble.infer.application.ports.driven.for_invoking_llm import ForInvokingLLM
from llm_ensemble.libs.logging.structlog_logger import get_logger


class DemoLLMAdapter(ForInvokingLLM):
    """Demo LLM provider that replays real responses from a lookup file.

    Matches prompts by SHA256 content hash to return the exact response
    that was produced by a real LLM for that prompt.
    """

    VERSION = "1.0"
    PROVIDER_NAME = "demo"

    def __init__(
        self,
        model_config: ModelConfig,
        retry_config: RetryConfig,
    ):
        self.model_config = model_config
        self.retry_config = retry_config
        self.logger = get_logger(component=f"{self.PROVIDER_NAME}_provider")
        self._call_count = 0

        # Resolve responses file: check additional_params for custom path,
        # otherwise use the default
        configs_dir = Path(__file__).resolve().parents[6] / "configs"
        custom_file = (model_config.additional_params or {}).get("responses_file")
        if custom_file:
            responses_path = configs_dir / custom_file
        else:
            responses_path = configs_dir / "demo_responses.json"

        # Load prompt_hash → response lookup
        if not responses_path.exists():
            raise FileNotFoundError(
                f"Responses file not found: {responses_path}\n"
                "Generate it from the backup database first."
            )
        with open(responses_path) as f:
            self._responses: dict[str, str] = json.load(f)

        self.logger.info(
            "demo_adapter_loaded",
            responses_count=len(self._responses),
            responses_path=str(responses_path),
        )

    def get_provider(self):
        from llm_ensemble.infer.domain.entities.provider import Provider

        return Provider(name=self.PROVIDER_NAME, version=self.VERSION)

    def get_model_config(self) -> ModelConfig:
        return self.model_config

    def get_retry_config(self) -> RetryConfig:
        return self.retry_config

    def infer(
        self,
        prompt: str,
    ) -> tuple[str, LLMInvocationMetrics]:
        """Look up real response by prompt hash, with simulated latency.

        Args:
            prompt: Pre-built prompt string

        Returns:
            Tuple of (response_text, invocation_metrics)

        Raises:
            KeyError: If no response found for this prompt hash
        """
        # Simulate API latency
        latency_ms = self._simulate_latency()

        # Hash the prompt the same way the DB does (SHA256)
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()

        # Look up real response
        response_text = self._responses.get(prompt_hash)
        if response_text is None:
            raise KeyError(
                f"No response found for prompt hash: {prompt_hash[:16]}... "
                f"({len(self._responses)} responses loaded). "
                "Ensure the responses file was generated from the same "
                "ingest data and prompt template."
            )

        # Estimate token counts
        prompt_tokens = max(1, len(prompt) // 4)
        completion_tokens = max(1, len(response_text) // 4)
        total_tokens = prompt_tokens + completion_tokens

        # Calculate costs if pricing is available
        cost_estimate_usd = None
        actual_cost_usd = None
        if self.model_config.pricing:
            prompt_cost = (
                prompt_tokens / 1_000_000
            ) * self.model_config.pricing.prompt_cost_per_1m_tokens
            completion_cost = (
                completion_tokens / 1_000_000
            ) * self.model_config.pricing.completion_cost_per_1m_tokens
            cost_estimate_usd = prompt_cost + completion_cost
            actual_cost_usd = cost_estimate_usd

        metrics = LLMInvocationMetrics(
            latency_ms=latency_ms,
            retries=0,
            cost_estimate_usd=cost_estimate_usd,
            actual_cost_usd=actual_cost_usd,
            generation_id=f"gen-{self._call_count}",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

        self._call_count += 1
        return response_text, metrics

    def _simulate_latency(self) -> float:
        """Simulate realistic API latency."""
        latency_ms = random.uniform(1000.0, 2000.0)
        time.sleep(latency_ms / 1000.0)
        return latency_ms
