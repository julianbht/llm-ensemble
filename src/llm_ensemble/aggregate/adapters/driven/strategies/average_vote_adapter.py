"""Average vote aggregation strategy adapter.

Averages numeric relevance scores and rounds to nearest label.
Confidence reflects agreement among models (inverse of normalized std deviation).
"""

from __future__ import annotations
import statistics

from llm_ensemble.infer.domain.entities.llm_judgement import LLMJudgement
from llm_ensemble.aggregate.application.ports.driven.for_aggregating import ForAggregating
from llm_ensemble.aggregate.domain.entities.aggregated_vote import AggregatedVote
from llm_ensemble.aggregate.domain.entities.aggregation_strategy import AggregationStrategy
from llm_ensemble.aggregate.domain.aggregated_vote_builder import build_aggregated_vote
from llm_ensemble.libs.schemas.relevance_score import RelevanceScore


class AverageVoteAdapter(ForAggregating):
    """Average vote aggregation strategy adapter.

    Calculates mean of numeric relevance scores and rounds to nearest label.

    Confidence calculation: Based on inverse of normalized standard deviation.
    Perfect agreement (std=0) yields confidence=1.0, maximum disagreement yields lower confidence.

    Strategy identity comes from config via constructor.
    """

    # Theoretical maximum standard deviation for RelevanceScore (0-3 scale)
    # Occurs when half votes are 0 and half are 3: std_dev ≈ 1.5
    MAX_STD_DEV = 1.5

    def __init__(self, strategy_name: str):
        """Initialize average vote adapter with strategy name.

        Args:
            strategy_name: Natural key for AggregationStrategy entity (from config)
        """
        self.strategy_name = strategy_name

    def aggregate(self, judgements: list[LLMJudgement]) -> AggregatedVote:
        """Apply average vote logic and create AggregatedVote entity.

        Args:
            judgements: All model judgements for a single (query_id, docid) pair

        Returns:
            AggregatedVote domain entity with consensus decision and metadata
        """
        # Extract valid numeric scores for averaging
        valid_scores: list[int] = []

        for judgement in judgements:
            # Get numeric score from llm_score.label (if available)
            if judgement.llm_score is not None and judgement.llm_score.label is not None:
                valid_scores.append(judgement.llm_score.label.value)

        # Handle edge case: no valid votes
        if not valid_scores:
            return build_aggregated_vote(
                llm_judgements=judgements,
                final_label=None,
                final_confidence=0.0,
                final_reasoning="No valid votes (all models failed to parse)",
            )

        # Calculate mean score
        mean_score = statistics.mean(valid_scores)

        # Round to nearest integer to get final label
        rounded_score = round(mean_score)

        # Clamp to valid range [0, 3]
        final_score = max(0, min(3, rounded_score))

        # Convert numeric score to RelevanceScore enum
        final_label = RelevanceScore(final_score)

        # Calculate confidence based on agreement (inverse of normalized std deviation)
        if len(valid_scores) == 1:
            # Single vote: perfect confidence
            final_confidence = 1.0
            std_dev = 0.0
        else:
            std_dev = statistics.stdev(valid_scores)
            # Normalize std_dev and invert: 0 std_dev = 1.0 confidence, max std_dev = lower confidence
            normalized_std = std_dev / self.MAX_STD_DEV
            # Clamp to [0, 1] range in case actual std_dev exceeds MAX_STD_DEV
            final_confidence = max(0.0, min(1.0, 1.0 - normalized_std))

        # Build reasoning string
        total_votes = len(valid_scores)
        reasoning = (
            f"Average score: {mean_score:.2f} (from {total_votes} models) "
            f"→ {final_label.label} (rounded to {final_score})"
        )

        if total_votes > 1:
            reasoning += f", std_dev={std_dev:.2f}"

        # Build AggregatedVote entity
        return build_aggregated_vote(
            llm_judgements=judgements,
            final_label=final_label,
            final_confidence=final_confidence,
            final_reasoning=reasoning,
        )

    def get_strategy(self) -> AggregationStrategy:
        """Get AggregationStrategy metadata for this adapter.

        Returns:
            AggregationStrategy entity with id and name
        """
        return AggregationStrategy(name=self.strategy_name)
