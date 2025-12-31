"""Domain logic for building AggregatedVote entities.

Separates construction logic (domain logic) from data structure (entity).
"""

from __future__ import annotations
from typing import Optional

from llm_ensemble.infer.domain.entities.llm_judgement import LLMJudgement
from llm_ensemble.aggregate.domain.entities.aggregated_vote import AggregatedVote
from llm_ensemble.libs.schemas import RelevanceScore


def build_aggregated_vote(
    llm_judgements: list[LLMJudgement],
    final_label: Optional[RelevanceScore] = None,
    final_confidence: Optional[float] = None,
    final_reasoning: Optional[str] = None,
) -> AggregatedVote:
    """Build AggregatedVote from domain entities.

    Note: Assumes all judgements are for the same dataset_sample.
    This invariant is guaranteed by the application layer's grouping logic
    and validated once at the run level via validate_infer_run_outputs_for_aggregation().

    Args:
        llm_judgements: All LLM judgements that were aggregated (for same dataset_sample)
        final_label: Consensus label chosen by the strategy
        final_confidence: Confidence in the aggregated decision
        final_reasoning: Explanation of how consensus was reached

    Returns:
        AggregatedVote with random UUID
    """
    return AggregatedVote(
        llm_judgements=llm_judgements,
        final_label=final_label,
        final_confidence=final_confidence,
        final_reasoning=final_reasoning,
    )
