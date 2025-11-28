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
from uuid import UUID
from pydantic import BaseModel, Field

from llm_ensemble.ingest.schemas.dataset_sample import DatasetSample
from llm_ensemble.infer.schemas.warnings import BaseWarning
from llm_ensemble.libs.schemas import RelevanceScore
from llm_ensemble.libs.db import (
    compute_llm_prompt_text_uuid,
    compute_llm_invocation_metrics_uuid,
    compute_llm_response_text_uuid,
    compute_llm_score_uuid,
    compute_llm_judgement_uuid,
)


class LLMPrompt(BaseModel):
    """Rendered prompt text sent to LLM, built from a dataset sample.

    Captures the semantic relationship: prompt is built FROM dataset_sample.
    UUID is computed from (prompt_template_id, dataset_sample_id, prompt_text).
    """

    id: UUID = Field(
        ...,
        description="Deterministic UUID computed from natural key"
    )

    dataset_sample: DatasetSample = Field(
        ...,
        description="The dataset sample this prompt was built from"
    )

    prompt_text: str = Field(
        ...,
        description="The rendered prompt text sent to the LLM"
    )

    @classmethod
    def create(
        cls,
        dataset_sample: DatasetSample,
        prompt_text: str,
        prompt_template_id: UUID
    ) -> "LLMPrompt":
        """Create an LLMPrompt with computed UUID.

        Args:
            dataset_sample: Dataset sample this prompt was built from
            prompt_text: Rendered prompt text
            prompt_template_id: Prompt template UUID (for UUID computation)

        Returns:
            LLMPrompt instance with computed ID
        """
        prompt_id = compute_llm_prompt_text_uuid(
            prompt_template_id=prompt_template_id,
            dataset_sample_id=dataset_sample.id,
            prompt_text=prompt_text
        )
        return cls(
            id=prompt_id,
            dataset_sample=dataset_sample,
            prompt_text=prompt_text
        )


class LLMInvocationMetrics(BaseModel):
    """Observability metrics from LLM API invocation.

    Captures performance and cost data from calling an LLM provider.
    UUID is computed from all metric fields.
    """

    id: UUID = Field(
        ...,
        description="Deterministic UUID computed from all metric fields"
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
        """Create LLMInvocationMetrics with computed UUID.

        Args:
            latency_ms: Inference time in milliseconds
            retries: Number of retries attempted
            cost_estimate_usd: Estimated cost in USD
            generation_id: Provider-specific generation ID
            prompt_tokens: Number of tokens in prompt
            completion_tokens: Number of tokens in completion
            total_tokens: Total tokens used

        Returns:
            LLMInvocationMetrics instance with computed ID
        """
        metrics_id = compute_llm_invocation_metrics_uuid(
            latency_ms=latency_ms,
            retries=retries,
            cost_estimate_usd=cost_estimate_usd,
            generation_id=generation_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
        return cls(
            id=metrics_id,
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

    Captures the semantic relationship: score is parsed FROM llm_response_text.
    This represents the structured score that a ResponseParser extracts
    from raw LLM output text. All fields are optional to handle parse failures.

    UUID is computed from (parser_spec_id, llm_response_text_id).
    """

    id: UUID = Field(
        ...,
        description="Deterministic UUID computed from parser_spec_id and llm_response_text"
    )

    llm_response_text: str = Field(
        ...,
        description="The raw LLM response text that was parsed to extract this score"
    )

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
        llm_response_text: str,
        parser_spec_id: UUID,
        label: Optional[RelevanceScore] = None,
        confidence: Optional[float] = None,
        rationale: Optional[str] = None,
        warnings: Optional[list[BaseWarning]] = None,
    ) -> "LLMScore":
        """Create an LLMScore with computed UUID.

        Args:
            llm_response_text: Raw LLM response text that was parsed
            parser_spec_id: Parser spec UUID (for UUID computation)
            label: Parsed relevance label
            confidence: Confidence score [0-1]
            rationale: Explanation for the judgement
            warnings: Parser warnings

        Returns:
            LLMScore instance with computed ID
        """
        llm_response_text_id = compute_llm_response_text_uuid(llm_response_text)
        score_id = compute_llm_score_uuid(parser_spec_id, llm_response_text_id)

        return cls(
            id=score_id,
            llm_response_text=llm_response_text,
            label=label,
            confidence=confidence,
            rationale=rationale,
            warnings=warnings or [],
        )


class LLMJudgement(BaseModel):
    """A complete LLM relevance judgement - pure domain model.

    Nested structure matching ORM semantics:
    - llm_prompt: Contains the dataset_sample (prompt built FROM sample)
    - invocation_metrics: Performance data from the LLM API call
    - llm_score: Contains the llm_response_text (score parsed FROM response)

    This captures the complete data lineage for a single inference:
    what was judged (in llm_prompt.dataset_sample), what prompt was sent,
    what response came back (in llm_score.llm_response_text), how the call
    performed, and what score was extracted.

    The structure mirrors the inference workflow:
    1. Build prompt from dataset sample → LLMPrompt (dataset_sample + prompt_text)
    2. Invoke LLM → response_text + LLMInvocationMetrics
    3. Parse response → LLMScore (llm_response_text + label/confidence/rationale)
    4. Create judgement

    UUID is computed from (judged_dataset_id, llm_prompt_id).
    Config IDs track which model and prompt were used for provenance.
    """

    id: UUID = Field(
        ...,
        description="Deterministic UUID computed from judged_dataset_id and llm_prompt_id"
    )

    model_config_id: UUID = Field(
        ...,
        description="Model configuration used for this judgement (for provenance)"
    )

    prompt_template_id: UUID = Field(
        ...,
        description="Prompt template used for this judgement (for provenance)"
    )

    llm_prompt: LLMPrompt = Field(
        ...,
        description="The prompt sent to the LLM (contains dataset_sample + prompt_text)"
    )

    invocation_metrics: LLMInvocationMetrics = Field(
        ...,
        description="Observability data from the LLM API call (latency, retries, cost, tokens)"
    )

    llm_score: Optional[LLMScore] = Field(
        None,
        description=(
            "The parsed score (contains llm_response_text + label/confidence/rationale). "
            "None if response parsing completely failed."
        )
    )

    @classmethod
    def create(
        cls,
        llm_prompt: LLMPrompt,
        invocation_metrics: LLMInvocationMetrics,
        model_config_id: UUID,
        prompt_template_id: UUID,
        judged_dataset_id: UUID,
        llm_score: Optional[LLMScore] = None,
    ) -> "LLMJudgement":
        """Create an LLMJudgement with computed UUID.

        Args:
            llm_prompt: Prompt sent to LLM (contains dataset_sample + prompt_text)
            invocation_metrics: Observability metrics
            model_config_id: Model configuration UUID (for provenance)
            prompt_template_id: Prompt template UUID (for provenance)
            judged_dataset_id: Judged dataset UUID (for UUID computation)
            llm_score: Parsed score (contains llm_response_text + parsed fields, None if parsing failed)

        Returns:
            LLMJudgement instance with computed ID
        """
        judgement_id = compute_llm_judgement_uuid(judged_dataset_id, llm_prompt.id)

        return cls(
            id=judgement_id,
            model_config_id=model_config_id,
            prompt_template_id=prompt_template_id,
            llm_prompt=llm_prompt,
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
