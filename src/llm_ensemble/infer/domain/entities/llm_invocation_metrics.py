"""LLMInvocationMetrics entity for the infer CLI.

Observability metrics from LLM API invocation.
"""

from __future__ import annotations
import uuid
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field


class LLMInvocationMetrics(BaseModel):
    """Observability metrics from LLM API invocation.

    Captures performance and cost data from calling an LLM provider.
    """

    id: UUID = Field(
        default_factory=uuid.uuid4,
        description="Random UUID for these metrics"
    )

    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Inference time in milliseconds"
    )

    retries: int = Field(
        0,
        ge=0,
        description="Number of retries attempted before success or failure"
    )

    cost_estimate_usd: Optional[float] = Field(
        None,
        ge=0.0,
        description="Estimated cost in USD for this inference call"
    )

    actual_cost_usd: Optional[float] = Field(
        None,
        ge=0.0,
        description="Actual cost in USD from provider's generation cost API"
    )

    generation_id: Optional[str] = Field(
        None,
        description="Provider-specific generation ID (e.g., OpenRouter gen-xxx) for async cost queries"
    )

    prompt_tokens: Optional[int] = Field(
        None,
        ge=0,
        description="Number of tokens in the prompt"
    )

    completion_tokens: Optional[int] = Field(
        None,
        ge=0,
        description="Number of tokens in the completion"
    )

    total_tokens: Optional[int] = Field(
        None,
        ge=0,
        description="Total tokens used (prompt + completion)"
    )
