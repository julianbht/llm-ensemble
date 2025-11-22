"""Inference workflow DTOs for the infer CLI.

This module contains all data structures used in the inference pipeline:
- LLMInvocationMetrics: Observability data from LLM API calls (latency, retries, cost, tokens)
- LLMScore: Parsed relevance assessment (label + confidence + rationale)
- LLMJudgement: Complete judgement combining prompt, response text, metrics, score, and sample

These are tightly coupled DTOs that represent the inference workflow pipeline.
They are components of the LLMJudgement aggregate root.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field

from llm_ensemble.ingest.schemas.judging_sample import JudgingSample
from llm_ensemble.infer.schemas.warnings import BaseWarning
from llm_ensemble.libs.schemas import RelevanceScore


class LLMInvocationMetrics(BaseModel):
    """Observability metrics from LLM API invocation.

    Captures performance and cost data from calling an LLM provider.
    These metrics are observed from the API call event and stored inline
    with the judgement record.
    """

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


class LLMScore(BaseModel):
    """Parsed relevance assessment extracted from LLM response.

    This represents the structured score that a ResponseParser extracts
    from raw LLM output text. All fields are optional to handle parse failures.

    If parsing completely fails, an LLMScore with all None fields can be created
    to represent "we got a response but couldn't parse it".
    """

    label: Optional[RelevanceScore] = Field(
        None,
        description=(
            "Parsed relevance label: "
            "0 = IRRELEVANT, 1 = RELEVANT, 2 = HIGHLY_RELEVANT, 3 = PERFECTLY_RELEVANT. "
            "None if parsing failed."
        )
    )

    confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="LLM self-reported or derived confidence score [0-1]. None if not available."
    )

    rationale: Optional[str] = Field(
        None,
        description="LLM's explanation for its relevance judgement. None if not parseable."
    )

    warnings: list[BaseWarning] = Field(
        default_factory=list,
        description="Parser-level warnings: parse errors, missing fields, validation issues, etc."
    )


class LLMJudgement(BaseModel):
    """A complete LLM relevance judgement - pure domain model.

    This is the canonical judgement schema that combines:
    - judging_sample: The input (query + document + gold score)
    - prompt: The rendered prompt text sent to the LLM
    - raw_response: The unparsed text returned by the LLM
    - invocation_metrics: Observability data (latency, retries, cost, tokens)
    - llm_score: The parsed relevance assessment (label + confidence + rationale)

    This captures the complete data lineage for a single inference:
    what was judged, what prompt was sent, what response came back,
    how the call performed, and what score was extracted.

    The structure mirrors the inference workflow:
    1. Build prompt from sample
    2. Invoke LLM (get raw_response + invocation_metrics)
    3. Parse response (get llm_score)
    4. Create judgement

    Note: Run context (model config, git SHA, etc.) is NOT part of the judgement itself.
    That metadata is provided separately to persistence adapters when needed.
    """

    judging_sample: JudgingSample = Field(
        ...,
        description="The input sample that was judged"
    )

    prompt: str = Field(
        ...,
        description="The rendered prompt text sent to the LLM for this inference"
    )

    raw_response: str = Field(
        ...,
        description="The unparsed text returned by the LLM"
    )

    invocation_metrics: LLMInvocationMetrics = Field(
        ...,
        description="Observability data from the LLM API call (latency, retries, cost, tokens)"
    )

    llm_score: Optional[LLMScore] = Field(
        None,
        description=(
            "The parsed relevance assessment (label + confidence + rationale). "
            "None if response parsing completely failed."
        )
    )

    def get_all_warnings(self) -> list[BaseWarning]:
        """Get all warnings from parsing stage.

        Returns parser warnings from llm_score (if score exists).

        Returns:
            List of parser warnings from this judgement
        """
        if self.llm_score is not None:
            return list(self.llm_score.warnings)

        return []
