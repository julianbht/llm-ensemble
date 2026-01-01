"""AggregatedVote - pure data entity for aggregation results.

Represents the output of a single aggregation strategy (e.g., majority vote, weighted vote)
applied to a group of LLM judgements for the same query-document pair.

Design:
- Pure data carrier - no business logic or factory methods
- Validation and construction logic in aggregated_vote_builder.py
- Stores full LLMJudgement objects for self-contained domain model
- Each LLMJudgement comes from a different judged_dataset (different model config)
"""

from __future__ import annotations
from typing import Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field

from llm_ensemble.libs.schemas.relevance_score import RelevanceScore
from llm_ensemble.infer.domain.entities.llm_judgement import LLMJudgement


class AggregatedVote(BaseModel):
    """Pure data entity for aggregation results.

    Contains the result from applying an ensemble strategy to multiple model predictions.

    Use `build_aggregated_vote()` from aggregated_vote_builder.py to create instances.
    """

    id: UUID = Field(
        default_factory=uuid4,
        description="Random UUID identifier"
    )

    llm_judgements: list[LLMJudgement] = Field(
        ...,
        description="All LLM judgements that were aggregated (one from each judged_dataset/model config, all for same dataset_sample)"
    )

    final_label: Optional[RelevanceScore] = Field(
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

    final_reasoning: Optional[str] = Field(
        None,
        description=(
            "Human-readable explanation of how consensus was reached. "
            "E.g., '3/5 models voted RELEVANT', 'tie broken by lowest label'"
        )
    )
