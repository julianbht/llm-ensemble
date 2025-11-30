"""DTO for LLM invocation results from provider adapters.

Simple data structure returned by provider adapters before domain object creation.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class LLMInvocationDTO(BaseModel):
    """Raw invocation result from LLM provider adapters.

    Adapters implement _do_infer_raw() and return this DTO.
    The port layer maps this to (response_text, LLMInvocationMetrics) tuple.
    """

    response_text: str = Field(
        ...,
        description="Raw response text from the LLM"
    )

    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Inference time in milliseconds"
    )

    cost_estimate_usd: Optional[float] = Field(
        None,
        ge=0.0,
        description="Estimated cost in USD for this inference call"
    )

    generation_id: Optional[str] = Field(
        None,
        description="Provider-specific generation ID for async cost queries"
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
