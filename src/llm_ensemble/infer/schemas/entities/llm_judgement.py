"""LLMJudgement entity for the infer CLI.

A complete LLM relevance judgement combining prompt, metrics, and score.
"""

from __future__ import annotations
import uuid
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field

from llm_ensemble.infer.schemas.entities.llm_prompt import LLMPrompt
from llm_ensemble.infer.schemas.entities.llm_invocation_metrics import LLMInvocationMetrics
from llm_ensemble.infer.schemas.entities.llm_score import LLMScore
from llm_ensemble.infer.schemas.entities.provider import Provider
from llm_ensemble.infer.schemas.model_config_schema import ModelConfig
from llm_ensemble.infer.schemas.warnings import BaseWarning


class LLMJudgement(BaseModel):
    """A complete LLM relevance judgement - pure domain model.

    Nested structure capturing complete context:
    - model_config: Full model configuration used for inference
    - provider: Which service/platform ran the inference
    - llm_prompt: Contains the dataset_sample and prompt_template
    - invocation_metrics: Performance data from the LLM API call
    - llm_score: Contains the llm_response_text and parser

    This captures the complete data lineage for a single inference:
    what was judged (in llm_prompt.dataset_sample), what prompt was sent,
    what model and provider were used, what response came back,
    how the call performed, and what score was extracted.

    The structure mirrors the inference workflow:
    1. Build prompt from dataset sample → LLMPrompt (with prompt_template)
    2. Invoke LLM → response_text + LLMInvocationMetrics
    3. Parse response → LLMScore (with parser)
    4. Create judgement with full context
    """

    id: UUID = Field(
        default_factory=uuid.uuid4,
        description="Random UUID for this judgement"
    )

    model_cfg: ModelConfig = Field(
        ...,
        description="Complete model configuration used for this judgement"
    )

    llm_provider: Provider = Field(
        ...,
        description="Provider/service that executed the inference"
    )

    llm_prompt: LLMPrompt = Field(
        ...,
        description="The prompt sent to the LLM (contains dataset_sample and prompt_template)"
    )

    llm_invocation_metrics: LLMInvocationMetrics = Field(
        ...,
        description="Observability data from the LLM API call (latency, retries, cost, tokens)"
    )

    llm_score: Optional[LLMScore] = Field(
        None,
        description=(
            "The parsed score (contains llm_response_text, parser, and label/confidence/rationale). "
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
