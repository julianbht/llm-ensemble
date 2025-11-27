"""AggregatedVote - result from applying an ensemble strategy to multiple model votes.

Represents the output of a single aggregation strategy (e.g., majority vote, weighted vote)
applied to a group of dataset_judgements for the same query-document pair.

This replaces the old AggregatedScore schema to match the new ORM structure.
"""

from __future__ import annotations
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field

from llm_ensemble.libs.schemas import RelevanceScore
from llm_ensemble.infer.schemas.dataset_judgement import DatasetJudgement


class AggregatedVote(BaseModel):
    """Result from applying an ensemble strategy to multiple model predictions.

    Contains:
    - id: Deterministic UUID from (dataset_vote_id, aggregation_spec_id)
    - dataset_vote_id: Which dataset_vote this aggregation belongs to
    - aggregation_spec_id: Which aggregation strategy was used
    - dataset_judgements: All dataset judgements that were aggregated (full objects)
    - final_label: Consensus label chosen by the strategy
    - final_confidence: Strategy's confidence in the decision
    - final_reasoning: Human-readable explanation of how consensus was reached

    Design: Stores full DatasetJudgement objects for self-contained domain model.
    Each DatasetJudgement contains LLMJudgements from different models.
    At persistence layer, AggregationVote ORM entities track FK relationships.
    """

    id: UUID = Field(
        ...,
        description="Deterministic UUID computed from dataset_vote_id and aggregation_spec_id"
    )

    dataset_vote_id: UUID = Field(
        ...,
        description="Which dataset_vote this aggregation belongs to"
    )

    aggregation_spec_id: UUID = Field(
        ...,
        description="Which aggregation strategy was used"
    )

    dataset_judgements: list[DatasetJudgement] = Field(
        default_factory=list,
        description="All dataset judgements that were aggregated (one from each run, each containing LLMJudgements)"
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
