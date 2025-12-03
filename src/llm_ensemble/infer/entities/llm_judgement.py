"""LLMJudgement entity for the infer CLI.

A complete LLM relevance judgement combining prompt, metrics, and score.
"""

from __future__ import annotations
import uuid
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field

from llm_ensemble.infer.entities.llm_prompt import LLMPrompt
from llm_ensemble.infer.entities.llm_invocation_metrics import LLMInvocationMetrics
from llm_ensemble.infer.entities.llm_score import LLMScore
from llm_ensemble.infer.entities.warnings import BaseWarning


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
