"""LLMResponse - LLM output from provider (without sample or manifest).

This is the data structure returned by LLM provider adapters.
The InferenceService will combine this with sample and manifest to create the final LLMJudgement.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field

from llm_ensemble.libs.schemas import RelevanceScore


class LLMResponse(BaseModel):
    """LLM response output from provider adapters.

    This represents just the LLM's output: the predicted score, rationale,
    and observability metadata. The domain service will combine this with
    the input sample and infer manifest to create the final LLMJudgement.

    This separation allows providers to focus on LLM inference without
    needing knowledge of the broader pipeline context (manifests, samples, etc.).
    """

    llm_score: Optional[RelevanceScore] = Field(
        None,
        description=(
            "LLM-predicted relevance score: "
            "0 = IRRELEVANT, 1 = RELEVANT, 2 = HIGHLY_RELEVANT, 3 = PERFECTLY_RELEVANT. "
            "None = parsing failed or model did not produce valid output."
        )
    )

    rationale: Optional[str] = Field(
        None,
        description="LLM's explanation for its relevance judgement"
    )

    confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="LLM self-reported or derived confidence score [0-1]"
    )

    raw_response: str = Field(
        ...,
        description="Unparsed LLM response text for debugging and transparency"
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

    warnings: list[str] = Field(
        default_factory=list,
        description="Parser warnings, API errors, fallbacks, validation issues, etc."
    )
