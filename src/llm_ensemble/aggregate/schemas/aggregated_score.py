"""AggregatedScore schema - result from applying an ensemble strategy to multiple model votes.

Represents the output of a single aggregation strategy (e.g., majority vote, weighted vote)
applied to a group of LLMJudgement predictions for the same query-document pair.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field

from llm_ensemble.libs.schemas import RelevanceScore


class AggregatedScore(BaseModel):
    """Result from applying an ensemble strategy to multiple model predictions.

    Contains:
    - final_relevance_score: Consensus label chosen by the strategy
    - final_confidence: Strategy's confidence in the decision
    - final_reasoning: Human-readable explanation of how consensus was reached

    Note: Individual model votes are NOT stored here. They can be derived from
    the constituent LLM calls via llm_call.response.label. For debugging which
    models voted what, use logging or separate analysis tools.
    """

    final_relevance_score: Optional[RelevanceScore] = Field(
        None,
        description="Consensus relevance label chosen by the strategy (None if no consensus possible)"
    )

    final_confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence in the aggregated decision [0-1]. "
            "For majority vote: fraction of votes for winning label. "
            "For weighted vote: sum of weights for winning label / total weights."
        )
    )

    final_reasoning: str = Field(
        default="",
        description=(
            "Human-readable explanation of how consensus was reached. "
            "E.g., '3/5 models voted RELEVANT', 'tie broken by lowest label'"
        )
    )
