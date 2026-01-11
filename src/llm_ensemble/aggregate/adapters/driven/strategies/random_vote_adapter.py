"""Majority vote with random tie-breaking aggregation strategy adapter.

Uses majority vote to find the most common label.
When there's a tie, randomly selects one of the tied labels instead of averaging.
"""

from __future__ import annotations
import random
from collections import Counter

from llm_ensemble.infer.domain.entities.llm_judgement import LLMJudgement
from llm_ensemble.aggregate.application.ports.driven.for_aggregating import (
    ForAggregating,
)
from llm_ensemble.aggregate.domain.entities.aggregated_vote import AggregatedVote
from llm_ensemble.aggregate.domain.entities.aggregation_strategy import (
    AggregationStrategy,
)
from llm_ensemble.aggregate.domain.aggregated_vote_builder import build_aggregated_vote
from llm_ensemble.libs.schemas.relevance_score import RelevanceScore


class RandomVoteAdapter(ForAggregating):
    """Majority vote with random tie-breaking aggregation strategy adapter.

    Counts votes for each label and selects the label with the most votes.

    Tie handling: If multiple labels have the same max count, randomly selects
    one of the tied labels (unlike majority_vote which averages tied labels).

    Confidence: Fraction of votes that went to the winning label.

    Uses fixed seed for reproducibility while maintaining uniform random distribution.

    Strategy identity comes from config via constructor.
    """

    RANDOM_SEED = 42

    def __init__(self, strategy_name: str):
        """Initialize random tie-breaking majority vote adapter with strategy name.

        Args:
            strategy_name: Natural key for AggregationStrategy entity (from config)
        """
        self.strategy_name = strategy_name
        self._random = random.Random(self.RANDOM_SEED)

    def aggregate(self, judgements: list[LLMJudgement]) -> AggregatedVote:
        """Apply majority vote with random tie-breaking logic and create AggregatedVote entity.

        Args:
            judgements: All model judgements for a single (query_id, docid) pair

        Returns:
            AggregatedVote domain entity with consensus decision and metadata
        """
        # Extract valid labels for voting
        valid_labels: list[RelevanceScore] = []

        for judgement in judgements:
            # Get label from llm_score (if available)
            if (
                judgement.llm_score is not None
                and judgement.llm_score.label is not None
            ):
                valid_labels.append(judgement.llm_score.label)

        # Handle edge case: no valid votes
        if not valid_labels:
            return build_aggregated_vote(
                llm_judgements=judgements,
                final_label=None,
                final_confidence=0.0,
                final_reasoning="No valid votes (all models failed to parse)",
            )

        # Count votes for each label
        vote_counts = Counter(valid_labels)

        # Find maximum vote count
        max_count = max(vote_counts.values())

        # Find all labels with max count (for tie detection)
        winners = [label for label, count in vote_counts.items() if count == max_count]

        # Break ties randomly: if multiple labels have same max count, pick one randomly
        if len(winners) > 1:
            final_label = self._random.choice(winners)
        else:
            final_label = winners[0]

        # Calculate confidence: fraction of votes for winning label
        total_votes = len(valid_labels)
        final_confidence = max_count / total_votes

        # Build reasoning string
        if len(winners) > 1:
            tied_labels_str = ", ".join(
                str(w.label) for w in sorted(winners, key=lambda x: x.value)
            )
            reasoning = (
                f"{max_count}/{total_votes} models each voted for: {tied_labels_str} "
                f"(tie broken randomly: selected {final_label.label})"
            )
        else:
            reasoning = f"{max_count}/{total_votes} models voted {final_label.label}"

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
