"""AggregatedScore schema - result from applying an ensemble strategy to multiple model votes.

Represents the output of a single aggregation strategy (e.g., majority vote, weighted vote)
applied to a group of LLMScore predictions for the same query-document pair.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field

from llm_ensemble.libs.schemas import RelevanceScore


class PerModelVote(BaseModel):
    """A single model's vote in the ensemble.
    
    Tracks which model contributed which prediction, enabling explainability
    and debugging of aggregation decisions.
    """
    
    model_id: str = Field(
        ...,
        description="Model identifier from the infer run (e.g., 'gpt-oss-20b')"
    )
    
    label: Optional[RelevanceScore] = Field(
        None,
        description="Model's predicted relevance label (None if model failed to parse)"
    )
    
    confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Model's self-reported confidence [0-1]"
    )


class AggregatedScore(BaseModel):
    """Result from applying a single ensemble strategy to multiple model predictions.
    
    Contains:
    - strategy: Which aggregation method was used
    - per_model_votes: All individual model predictions
    - final_relevance_score: Consensus label chosen by the strategy
    - final_confidence: Strategy's confidence in the decision
    - final_reasoning: Human-readable explanation of how consensus was reached
    
    Multiple strategies can be applied to the same input, producing multiple
    AggregatedScore objects per query-document pair.
    """
    
    strategy: str = Field(
        ...,
        description="Name of the aggregation strategy used (e.g., 'majority_vote', 'weighted_majority')"
    )
    
    per_model_votes: list[PerModelVote] = Field(
        ...,
        description="All individual model votes that were aggregated"
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
