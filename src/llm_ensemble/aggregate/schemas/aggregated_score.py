"""AggregatedScore schema - result from applying an ensemble strategy to multiple model votes.

Represents the output of a single aggregation strategy (e.g., majority vote, weighted vote)
applied to a group of LLMScore predictions for the same query-document pair.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field

from llm_ensemble.libs.schemas import RelevanceScore


class AggregatedScore(BaseModel):
    """Result from applying a single ensemble strategy to multiple model predictions.
    
    Contains:
    - strategy: Which aggregation method was used
    - per_model_votes: Individual model label values for debugging (matches order of judgements in AggregatedJudgement)
    - final_relevance_score: Consensus label chosen by the strategy
    - final_confidence: Strategy's confidence in the decision
    - final_reasoning: Human-readable explanation of how consensus was reached
    
    Multiple strategies can be applied to the same input, producing multiple
    AggregatedScore objects per query-document pair.
    
    Note: per_model_votes only stores label values for quick debugging inspection.
    Full model information (model_id, confidence, rationale, etc.) is available
    in the parent AggregatedJudgement.judgements list.
    """
    
    strategy: str = Field(
        ...,
        description="Name of the aggregation strategy used (e.g., 'majority_vote', 'weighted_majority')"
    )
    
    per_model_votes: list[Optional[int]] = Field(
        ...,
        description=(
            "Individual model label values in same order as AggregatedJudgement.judgements. "
            "Values are 0-3 for RelevanceScore, None if model failed to parse. "
            "For full model details (model_id, confidence, rationale), see parent judgements list."
        )
    )
    
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
