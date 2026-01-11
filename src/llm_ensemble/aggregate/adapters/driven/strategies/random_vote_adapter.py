"""Random vote aggregation strategy adapter.

Randomly selects one judgement from the available votes.
Confidence reflects the fraction of votes matching the randomly selected label.
"""

from __future__ import annotations
import random

from llm_ensemble.infer.domain.entities.llm_judgement import LLMJudgement
from llm_ensemble.aggregate.application.ports.driven.for_aggregating import ForAggregating
from llm_ensemble.aggregate.domain.entities.aggregated_vote import AggregatedVote
from llm_ensemble.aggregate.domain.entities.aggregation_strategy import AggregationStrategy
from llm_ensemble.aggregate.domain.aggregated_vote_builder import build_aggregated_vote
from llm_ensemble.libs.schemas.relevance_score import RelevanceScore


class RandomVoteAdapter(ForAggregating):
    """Random vote aggregation strategy adapter.

    Randomly selects one valid judgement from the available votes.

    Confidence calculation: Fraction of votes that match the randomly selected label.

    Uses fixed seed for reproducibility while maintaining uniform random distribution.

    Strategy identity comes from config via constructor.
    """

    RANDOM_SEED = 42

    def __init__(self, strategy_name: str):
        """Initialize random vote adapter with strategy name.

        Args:
            strategy_name: Natural key for AggregationStrategy entity (from config)
        """
        self.strategy_name = strategy_name
        self._random = random.Random(self.RANDOM_SEED)

    def aggregate(self, judgements: list[LLMJudgement]) -> AggregatedVote:
        """Apply random vote logic and create AggregatedVote entity.

        Args:
            judgements: All model judgements for a single (query_id, docid) pair

        Returns:
            AggregatedVote domain entity with consensus decision and metadata
        """
        # Extract valid labels for voting
        valid_labels: list[RelevanceScore] = []

        for judgement in judgements:
            # Get label from llm_score (if available)
            if judgement.llm_score is not None and judgement.llm_score.label is not None:
                valid_labels.append(judgement.llm_score.label)

        # Handle edge case: no valid votes
        if not valid_labels:
            return build_aggregated_vote(
                llm_judgements=judgements,
                final_label=None,
                final_confidence=0.0,
                final_reasoning="No valid votes (all models failed to parse)",
            )

        # Randomly select one label using seeded RNG for reproducibility
        final_label = self._random.choice(valid_labels)

        # Calculate confidence: fraction of votes that match the selected label
        matching_votes = sum(1 for label in valid_labels if label == final_label)
        total_votes = len(valid_labels)
        final_confidence = matching_votes / total_votes

        # Build reasoning string
        reasoning = (
            f"Randomly selected {final_label.label} from {total_votes} models "
            f"({matching_votes}/{total_votes} models voted {final_label.label})"
        )

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
