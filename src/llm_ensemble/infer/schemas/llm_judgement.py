"""Inference workflow DTOs for the infer CLI.

This module contains all data structures used in the inference pipeline:
- LLMRequest: What we send to the LLM (prompt + warnings)
- LLMResponse: Raw LLM output (unparsed text + observability metadata)
- LLMScore: Parsed relevance assessment (label + confidence + rationale)
- LLMJudgement: Complete judgement combining all above

These are tightly coupled DTOs that represent the inference workflow pipeline.
They are components of the LLMJudgement aggregate root.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field

from llm_ensemble.ingest.schemas.judging_sample import JudgingSample
from llm_ensemble.infer.schemas.infer_run_info import InferRunInfo
from llm_ensemble.infer.schemas.warnings import BaseWarning
from llm_ensemble.libs.schemas import RelevanceScore


class LLMRequest(BaseModel):
    """Request sent to LLM provider for inference.

    This represents what was sent to the LLM:
    - prompt: The rendered prompt text after template substitution
    - warnings: Issues encountered during prompt building (missing variables, rendering errors, etc.)

    The PromptBuilder adapter creates this by rendering templates with sample data.
    """

    prompt: str = Field(
        ...,
        description="The rendered prompt text sent to the LLM for this inference"
    )

    warnings: list[BaseWarning] = Field(
        default_factory=list,
        description="Prompt builder warnings: rendering errors, missing variables, validation issues, etc."
    )


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
    """A complete LLM relevance judgement with full provenance.

    This is the canonical judgement schema that combines:
    - sample: The input (query + document + gold score + ingest manifest)
    - llm_request: What we sent to the LLM (rendered prompt + prompt warnings)
    - llm_response: The raw LLM output (unparsed text + observability metadata + provider warnings)
    - llm_score: The parsed relevance assessment (label + confidence + rationale + parser warnings)
    - run_info: The inference run context (model config + prompt config + git info + input params)

    This captures the complete data lineage: what was judged, what prompt was sent,
    what response came back, and what score was extracted. Each stage has its own warnings.

    Many LLMJudgements share one InferRunInfo (Many-to-One relationship). The run_info
    contains immutable runtime context known before the run starts, allowing judgements
    to be serialized immediately without waiting for aggregate statistics.
    """

    judging_sample: JudgingSample = Field(
        ...,
        description="The input sample that was judged (includes ingest manifest)"
    )

    llm_request: LLMRequest = Field(
        ...,
        description="The request sent to the LLM (rendered prompt + prompt warnings)"
    )

    llm_response: LLMResponse = Field(
        ...,
        description="The raw LLM output (unparsed text + observability metadata + provider warnings)"
    )

    llm_score: Optional[LLMScore] = Field(
        None,
        description=(
            "The parsed relevance assessment (label + confidence + rationale). "
            "None if response parsing completely failed."
        )
    )

    run_info: InferRunInfo = Field(
        ...,
        description="Inference run context (Many-to-One: many judgements share one run_info)"
    )

    def get_all_warnings(self) -> list[BaseWarning]:
        """Aggregate all warnings from request, response, and score.

        Combines warnings from all three stages:
        - Prompt building (in llm_request)
        - LLM inference (in llm_response)
        - Response parsing (in llm_score)

        Returns:
            List of all warnings from this judgement (prompt + provider + parser)

        Example:
            >>> judgement = LLMJudgement(...)
            >>> all_warnings = judgement.get_all_warnings()
            >>> len(all_warnings)  # Total warnings from all stages
            5
        """
        warnings = []

        # Prompt warnings from llm_request
        warnings.extend(self.llm_request.warnings)

        # Provider warnings from llm_response
        warnings.extend(self.llm_response.warnings)

        # Parser warnings from llm_score (if score exists)
        if self.llm_score is not None:
            warnings.extend(self.llm_score.warnings)

        return warnings
