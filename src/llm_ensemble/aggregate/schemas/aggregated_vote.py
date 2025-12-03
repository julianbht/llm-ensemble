"""AggregatedVote - result from applying an ensemble strategy to multiple model votes.

Represents the output of a single aggregation strategy (e.g., majority vote, weighted vote)
applied to a group of LLM judgements for the same query-document pair.

This replaces the old AggregatedScore schema to match the new ORM structure.
"""

from __future__ import annotations
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field

from llm_ensemble.libs.schemas import RelevanceScore
from llm_ensemble.infer.schemas.entities.llm_judgement import LLMJudgement
from llm_ensemble.aggregate.schemas.aggregation_strategy import AggregationStrategy
from llm_ensemble.libs.db import compute_aggregated_vote_uuid


class AggregatedVote(BaseModel):
    """Result from applying an ensemble strategy to multiple model predictions.

    Contains:
    - id: Deterministic UUID from (dataset_sample_id, aggregation_strategy.id)
    - aggregation_strategy: Which aggregation strategy was used (full entity)
    - llm_judgements: All LLM judgements that were aggregated (full objects, all for same dataset_sample)
    - final_label: Consensus label chosen by the strategy
    - final_confidence: Strategy's confidence in the decision
    - final_reasoning: Human-readable explanation of how consensus was reached

    Design: Stores full LLMJudgement objects for self-contained domain model.
    Each LLMJudgement comes from a different judged_dataset (different model config).

    Globally unique: An AggregatedVote represents applying a specific strategy to
    judgements for a specific sample. It can belong to multiple AggregatedDatasets.

    Natural key: (dataset_sample_id, aggregation_strategy.id)
    where dataset_sample_id is extracted from llm_judgements[0].llm_prompt.dataset_sample.id
    """

    id: UUID = Field(
        ...,
        description="Deterministic UUID computed from dataset_sample_id and aggregation_strategy.id"
    )

    aggregation_strategy: AggregationStrategy = Field(
        ...,
        description="Which aggregation strategy was used (full entity: id + name)"
    )

    llm_judgements: list[LLMJudgement] = Field(
        default_factory=list,
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

    @classmethod
    def create(
        cls,
        aggregation_strategy: AggregationStrategy,
        llm_judgements: list[LLMJudgement],
        final_label: Optional[RelevanceScore] = None,
        final_confidence: Optional[float] = None,
        final_reasoning: Optional[str] = None,
    ) -> "AggregatedVote":
        """Create AggregatedVote with computed ID.

        Args:
            aggregation_strategy: Which aggregation strategy was used (full entity)
            llm_judgements: All LLM judgements that were aggregated (must all be for same dataset_sample)
            final_label: Consensus label chosen by the strategy
            final_confidence: Confidence in the aggregated decision
            final_reasoning: Explanation of how consensus was reached

        Returns:
            AggregatedVote with deterministic UUID

        Raises:
            ValueError: If llm_judgements is empty or judgements have different dataset_sample_ids
        """
        if not llm_judgements:
            raise ValueError("llm_judgements cannot be empty")

        # Extract dataset_sample_id from first judgement
        dataset_sample_id = llm_judgements[0].llm_prompt.dataset_sample.id

        # Validate all judgements are for the same sample
        for judgement in llm_judgements[1:]:
            if judgement.llm_prompt.dataset_sample.id != dataset_sample_id:
                raise ValueError(
                    f"All llm_judgements must be for the same dataset_sample. "
                    f"Expected {dataset_sample_id}, got {judgement.llm_prompt.dataset_sample.id}"
                )

        # Compute deterministic UUID from dataset_sample_id and aggregation_strategy.id
        aggregated_vote_id = compute_aggregated_vote_uuid(
            dataset_sample_id,
            aggregation_strategy.id
        )

        return cls(
            id=aggregated_vote_id,
            aggregation_strategy=aggregation_strategy,
            llm_judgements=llm_judgements,
            final_label=final_label,
            final_confidence=final_confidence,
            final_reasoning=final_reasoning,
        )
