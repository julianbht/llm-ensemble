"""Retry configuration schema for LLM provider inference.

Defines retry behavior for API calls to LLM providers, including exponential
backoff parameters, retryable error codes, and request timeouts.

Example YAML config (configs/retries/standard.yaml):
    max_retries: 5
    base_delay_seconds: 1.0
    max_delay_seconds: 60.0
    request_timeout_seconds: 120
    retryable_status_codes: [429, 503, 504]
"""

from __future__ import annotations
from pydantic import Field

from llm_ensemble.libs.schemas.base_config import BaseConfig


class RetryConfig(BaseConfig):
    """Retry configuration for LLM provider API calls.

    Defines exponential backoff behavior, request timeouts, and which HTTP
    status codes should trigger retries.

    Attributes:
        max_retries: Maximum number of retry attempts (default: 5)
        base_delay_seconds: Initial delay before first retry (default: 1.0s)
        max_delay_seconds: Maximum delay cap for exponential backoff (default: 60.0s)
        request_timeout_seconds: Maximum time to wait for a single API request (default: 120s)
        retryable_status_codes: HTTP status codes that trigger retries (default: [429, 503, 504])
    """

    max_retries: int = Field(
        default=5,
        ge=0,
        description="Maximum number of retry attempts before failing"
    )

    base_delay_seconds: float = Field(
        default=1.0,
        gt=0.0,
        description="Initial delay in seconds before first retry (exponential backoff base)"
    )

    max_delay_seconds: float = Field(
        default=60.0,
        gt=0.0,
        description="Maximum delay cap in seconds for exponential backoff"
    )

    request_timeout_seconds: int = Field(
        default=120,
        gt=0,
        description="Maximum time in seconds to wait for a single API request before timing out"
    )

    retryable_status_codes: list[int] = Field(
        default=[429, 503, 504],
        description="HTTP status codes that should trigger retries (e.g., rate limits, server errors)"
    )
