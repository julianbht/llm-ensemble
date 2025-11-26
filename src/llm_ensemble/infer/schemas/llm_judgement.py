"""Inference workflow DTOs for the infer CLI.

This module contains all data structures used in the inference pipeline:
- LLMPrompt: Prompt text sent to LLM
- LLMResponse: Raw LLM response text
- LLMInvocationMetrics: Observability data from LLM API calls
- LLMScore: Parsed relevance assessment
- LLMJudgement: Complete judgement combining all components

These are pure domain entities (DTOs) used for data transfer and validation.
The ORM models for persistence are separate (see orms_normalized.py).
UUID computation is handled by mappers during ORM conversion.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field

from llm_ensemble.ingest.schemas.judging_sample import JudgingSample
from llm_ensemble.infer.schemas.warnings import BaseWarning
from llm_ensemble.libs.schemas import RelevanceScore


class LLMPrompt(BaseModel):
    """Rendered prompt text sent to LLM.

    Pure domain entity without persistence concerns.
    UUID computation handled by mappers during ORM conversion.
    """

    prompt_text: str = Field(
        ...,
        description="The rendered prompt text sent to the LLM"
    )

    @classmethod
    def create(cls, prompt_text: str) -> "LLMPrompt":
        """Create an LLMPrompt.

        Args:
            prompt_text: Rendered prompt text

        Returns:
            LLMPrompt instance
        """
        return cls(prompt_text=prompt_text)


class LLMResponse(BaseModel):
    """Raw response text from LLM.

    Pure domain entity without persistence concerns.
    UUID computation handled by mappers during ORM conversion.
    """

    raw_response: str = Field(
        ...,
        description="The unparsed text returned by the LLM"
    )

    @classmethod
    def create(cls, raw_response: str) -> "LLMResponse":
        """Create an LLMResponse.

        Args:
            raw_response: Raw LLM response text

        Returns:
            LLMResponse instance
        """
        return cls(raw_response=raw_response)


class LLMInvocationMetrics(BaseModel):
    """Observability metrics from LLM API invocation.

    Captures performance and cost data from calling an LLM provider.
    Pure domain entity without persistence concerns.
    UUID computation handled by mappers during ORM conversion.
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

    @classmethod
    def create(
        cls,
        latency_ms: float,
        retries: int = 0,
        cost_estimate_usd: Optional[float] = None,
        generation_id: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
    ) -> "LLMInvocationMetrics":
        """Create LLMInvocationMetrics.

        Args:
            latency_ms: Inference time in milliseconds
            retries: Number of retries attempted
            cost_estimate_usd: Estimated cost in USD
            generation_id: Provider-specific generation ID
            prompt_tokens: Number of tokens in prompt
            completion_tokens: Number of tokens in completion
            total_tokens: Total tokens used

        Returns:
            LLMInvocationMetrics instance
        """
        return cls(
            latency_ms=latency_ms,
            retries=retries,
            cost_estimate_usd=cost_estimate_usd,
            generation_id=generation_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )


class LLMScore(BaseModel):
    """Parsed relevance assessment extracted from LLM response.

    This represents the structured score that a ResponseParser extracts
    from raw LLM output text. All fields are optional to handle parse failures.

    Pure domain entity without persistence concerns.
    UUID computation handled by mappers during ORM conversion.
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

    @classmethod
    def create(
        cls,
        label: Optional[RelevanceScore] = None,
        confidence: Optional[float] = None,
        rationale: Optional[str] = None,
        warnings: Optional[list[BaseWarning]] = None,
    ) -> "LLMScore":
        """Create an LLMScore.

        Args:
            label: Parsed relevance label
            confidence: Confidence score [0-1]
            rationale: Explanation for the judgement
            warnings: Parser warnings

        Returns:
            LLMScore instance
        """
        return cls(
            label=label,
            confidence=confidence,
            rationale=rationale,
            warnings=warnings or [],
        )


class LLMJudgement(BaseModel):
    """A complete LLM relevance judgement - pure domain model.

    This is the canonical judgement schema that combines:
    - judging_sample: The input (query + document + gold score)
    - llm_prompt: The rendered prompt text sent to the LLM
    - llm_response: The unparsed text returned by the LLM
    - invocation_metrics: Observability data (latency, retries, cost, tokens)
    - llm_score: The parsed relevance assessment (label + confidence + rationale)

    This captures the complete data lineage for a single inference:
    what was judged, what prompt was sent, what response came back,
    how the call performed, and what score was extracted.

    The structure mirrors the inference workflow:
    1. Build prompt from sample (LLMPrompt)
    2. Invoke LLM (LLMResponse + LLMInvocationMetrics)
    3. Parse response (LLMScore)
    4. Create judgement

    Pure domain entity without persistence concerns.
    UUID computation handled by mappers during ORM conversion.
    """

    judging_sample: JudgingSample = Field(
        ...,
        description="The input sample that was judged"
    )

    llm_prompt: LLMPrompt = Field(
        ...,
        description="The rendered prompt text sent to the LLM"
    )

    llm_response: LLMResponse = Field(
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

    @classmethod
    def create(
        cls,
        judging_sample: JudgingSample,
        llm_prompt: LLMPrompt,
        llm_response: LLMResponse,
        invocation_metrics: LLMInvocationMetrics,
        llm_score: Optional[LLMScore] = None,
    ) -> "LLMJudgement":
        """Create an LLMJudgement.

        Args:
            judging_sample: Input sample that was judged
            llm_prompt: Rendered prompt sent to LLM
            llm_response: Raw response from LLM
            invocation_metrics: Observability metrics
            llm_score: Parsed score (None if parsing failed)

        Returns:
            LLMJudgement instance
        """
        return cls(
            judging_sample=judging_sample,
            llm_prompt=llm_prompt,
            llm_response=llm_response,
            invocation_metrics=invocation_metrics,
            llm_score=llm_score,
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
