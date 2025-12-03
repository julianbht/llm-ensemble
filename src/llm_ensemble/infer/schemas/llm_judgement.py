"""Inference workflow DTOs for the infer CLI.

This module contains all data structures used in the inference pipeline:
- LLMPrompt: Prompt text sent to LLM
- LLMResponse: Raw LLM response text
- LLMInvocationMetrics: Observability data from LLM API calls
- LLMScore: Parsed relevance assessment
- LLMJudgement: Complete judgement combining all components

These are pure domain entities (DTOs) used for data transfer and validation.
The ORM models for persistence are separate (see orms_normalized.py).
"""

from __future__ import annotations
import uuid
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field

from llm_ensemble.ingest.schemas.dataset_sample import DatasetSample
from llm_ensemble.infer.schemas.warnings import BaseWarning
from llm_ensemble.libs.schemas import RelevanceScore


class LLMPrompt(BaseModel):
    """Rendered prompt text sent to LLM, built from a dataset sample.

    Captures the semantic relationship: prompt is built FROM dataset_sample.
    """

    id: UUID = Field(
        default_factory=uuid.uuid4,
        description="Random UUID for this prompt"
    )

    dataset_sample: DatasetSample = Field(
        ...,
        description="The dataset sample this prompt was built from"
    )

    prompt_text: str = Field(
        ...,
        description="The rendered prompt text sent to the LLM"
    )


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

    Captures the semantic relationship: score is parsed FROM llm_response_text.
    This represents the structured score that a ResponseParser extracts
    from raw LLM output text. All fields are optional to handle parse failures.
    """

    id: UUID = Field(
        default_factory=uuid.uuid4,
        description="Random UUID for this score"
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

    Config IDs track which model and prompt were used for provenance.
    """

    id: UUID = Field(
        default_factory=uuid.uuid4,
        description="Random UUID for this judgement"
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

    def get_all_warnings(self) -> list[BaseWarning]:
        """Get all warnings from parsing stage.

        Returns parser warnings from llm_score (if score exists).

        Returns:
            List of parser warnings from this judgement
        """
        if self.llm_score is not None:
            return list(self.llm_score.warnings)

        return []
