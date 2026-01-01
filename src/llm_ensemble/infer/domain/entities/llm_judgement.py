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
from llm_ensemble.infer.domain.entities.llm_prompt_text import LLMPromptText
from llm_ensemble.infer.domain.entities.llm_response_text import LLMResponseText


class LLMJudgement(BaseModel):
    """A complete LLM relevance judgement - pure domain model.

    Mirrors LLMJudgementORM structure with embedded objects (not IDs):
    - dataset_sample: The query-document pair being judged
    - llm_prompt_text: The deduplicated prompt text entity sent to the LLM
    - llm_response_text: The deduplicated response text entity from the LLM
    - llm_invocation_metrics: Performance data (latency, cost, tokens)
    - llm_score: Parsed score (label, confidence, rationale, parse_issues)

    Configurations (ModelConfig, AdapterConfig with PromptBuilder/Parser/Provider)
    are NOT stored here - they live at the JudgedDataset level.

    This captures the complete data for a single inference:
    - What was judged (dataset_sample)
    - What prompt was sent (llm_prompt_text)
    - What response came back (llm_response_text)
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

    llm_prompt_text: LLMPromptText = Field(
        ...,
        description="The deduplicated prompt text entity sent to the LLM"
    )

    llm_response_text: LLMResponseText = Field(
        ...,
        description="The deduplicated response text entity from the LLM"
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

    parser_issue: Optional[ParserIssue] = Field(
        default=None,
        description="Primary parser issue if parsing encountered problems, None if clean parse."
    )
