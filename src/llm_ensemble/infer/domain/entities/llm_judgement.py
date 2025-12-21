"""LLMJudgement entity and builder for the infer CLI.

A complete LLM relevance judgement - mirrors LLMJudgementORM structure.
Configs stored at JudgedDataset level, not on individual judgements.
"""

from __future__ import annotations
import uuid
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field

from llm_ensemble.ingest.schemas.dataset_sample import DatasetSample
from llm_ensemble.infer.domain.entities.llm_invocation_metrics import LLMInvocationMetrics
from llm_ensemble.infer.domain.entities.llm_score import LLMScore
from llm_ensemble.infer.schemas.warnings import BaseWarning


class LLMJudgement(BaseModel):
    """A complete LLM relevance judgement - pure domain model.

    Mirrors LLMJudgementORM structure with embedded objects (not IDs):
    - dataset_sample: The query-document pair being judged
    - prompt_text: The rendered prompt sent to the LLM
    - response_text: The raw LLM response
    - llm_invocation_metrics: Performance data (latency, cost, tokens)
    - llm_score: Parsed score (label, confidence, rationale, warnings)

    Configurations (ModelConfig, AdapterConfig with PromptBuilder/Parser/Provider)
    are NOT stored here - they live at the JudgedDataset level.

    This captures the complete data for a single inference:
    - What was judged (dataset_sample)
    - What prompt was sent (prompt_text)
    - What response came back (response_text)
    - How the call performed (llm_invocation_metrics)
    - What score was extracted (llm_score)
    """

    id: UUID = Field(
        default_factory=uuid.uuid4,
        description="Random UUID for this judgement"
    )

    dataset_sample: DatasetSample = Field(
        ...,
        description="The query-document pair being judged"
    )

    prompt_text: str = Field(
        ...,
        description="The rendered prompt text sent to the LLM"
    )

    response_text: str = Field(
        ...,
        description="The raw LLM response text"
    )

    llm_invocation_metrics: LLMInvocationMetrics = Field(
        ...,
        description="Observability data from the LLM API call (latency, retries, cost, tokens)"
    )

    llm_score: Optional[LLMScore] = Field(
        None,
        description=(
            "The parsed score (label/confidence/rationale). "
            "None if response parsing completely failed."
        )
    )

    parser_warnings: list[BaseWarning] = Field(
        default_factory=list,
        description="Parser-level warnings: parse errors, missing fields, validation issues, etc."
    )


class LLMJudgementBuilder:
    """Builder for LLMJudgement that mirrors the inference pipeline flow.

    Incrementally builds a judgement as each step of inference completes:
    1. Start with the dataset sample being judged
    2. Add the prompt after building it
    3. Add the LLM response and metrics after inference
    4. Add the parsed score and warnings after parsing
    5. Build the final judgement

    This makes the pipeline flow explicit in the code.
    """

    def __init__(self, dataset_sample: DatasetSample):
        """Initialize builder with the sample being judged.

        Args:
            dataset_sample: The query-document pair being judged
        """
        self._dataset_sample = dataset_sample
        self._prompt_text: Optional[str] = None
        self._response_text: Optional[str] = None
        self._llm_invocation_metrics: Optional[LLMInvocationMetrics] = None
        self._llm_score: Optional[LLMScore] = None
        self._parser_warnings: list[BaseWarning] = []

    def with_prompt(self, prompt_text: str) -> "LLMJudgementBuilder":
        """Add the prompt text (called after prompt building).

        Args:
            prompt_text: The rendered prompt sent to the LLM

        Returns:
            Self for method chaining
        """
        self._prompt_text = prompt_text
        return self

    def with_llm_response(
        self,
        response_text: str,
        invocation_metrics: LLMInvocationMetrics
    ) -> "LLMJudgementBuilder":
        """Add the LLM response and metrics (called after inference).

        Args:
            response_text: Raw response text from the LLM
            invocation_metrics: Performance metrics (latency, cost, tokens)

        Returns:
            Self for method chaining
        """
        self._response_text = response_text
        self._llm_invocation_metrics = invocation_metrics
        return self

    def with_parsed_score(
        self,
        llm_score: Optional[LLMScore],
        parser_warnings: list[BaseWarning]
    ) -> "LLMJudgementBuilder":
        """Add the parsed score and warnings (called after parsing).

        Args:
            llm_score: Parsed score (may be None if parsing failed)
            parser_warnings: Any warnings from the parsing process

        Returns:
            Self for method chaining
        """
        self._llm_score = llm_score
        self._parser_warnings = parser_warnings
        return self

    def build(self) -> LLMJudgement:
        """Build the final LLMJudgement entity.

        Returns:
            Complete LLMJudgement entity

        Raises:
            ValueError: If required fields are missing
        """
        if self._prompt_text is None:
            raise ValueError("prompt_text is required - call with_prompt() first")
        if self._response_text is None:
            raise ValueError("response_text is required - call with_llm_response() first")
        if self._llm_invocation_metrics is None:
            raise ValueError("llm_invocation_metrics is required - call with_llm_response() first")

        return LLMJudgement(
            dataset_sample=self._dataset_sample,
            prompt_text=self._prompt_text,
            response_text=self._response_text,
            llm_invocation_metrics=self._llm_invocation_metrics,
            llm_score=self._llm_score,
            parser_warnings=self._parser_warnings,
        )
