"""LLMJudgement entity for the infer CLI.

A complete LLM relevance judgement - mirrors LLMJudgementORM structure.
Configs stored at JudgedDataset level, not on individual judgements.
"""

from __future__ import annotations
import uuid
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field

from llm_ensemble.ingest.domain.entities.dataset_sample import DatasetSample
from llm_ensemble.infer.domain.entities.llm_invocation_metrics import LLMInvocationMetrics
from llm_ensemble.infer.domain.entities.llm_score import LLMScore
from llm_ensemble.infer.domain.entities.parse_issues import ParserIssue


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

    parse_issues: list[ParserIssue] = Field(
        default_factory=list,
        description="Parser-level warnings: parse errors, missing fields, validation issues, etc."
    )
