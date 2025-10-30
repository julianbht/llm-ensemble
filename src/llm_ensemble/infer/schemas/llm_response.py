"""LLMResponse - Raw LLM output from provider (unparsed).

This is the data structure returned by LLM provider adapters.
It contains ONLY raw response text and observability metadata.
The ResponseParser will extract structured LLMScore from the raw_response text.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field

from llm_ensemble.infer.schemas.warnings import BaseWarning


class LLMResponse(BaseModel):
    """Raw LLM response output from provider adapters.

    This represents the raw output from calling an LLM API:
    - raw_response: The unparsed text returned by the model
    - Observability metadata: latency, retries, cost, warnings

    This schema contains NO parsed/structured data. The ResponseParser
    is responsible for extracting structured LLMScore from raw_response.

    The domain service coordinates: Provider returns LLMResponse →
    Parser extracts LLMScore → Service combines into LLMJudgement.
    """

    raw_response: str = Field(
        ...,
        description="Unparsed LLM response text (will be parsed by ResponseParser)"
    )

    # Observability metadata
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

    warnings: list[BaseWarning] = Field(
        default_factory=list,
        description="Provider-level warnings: API errors, fallbacks, network issues, etc."
    )
