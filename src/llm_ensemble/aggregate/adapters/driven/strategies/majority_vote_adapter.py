"""Majority vote aggregation strategy adapter.

Simple majority vote: the label with the most votes wins.
Ties are broken deterministically by choosing the lowest numeric label.
"""

from __future__ import annotations
from collections import Counter

from llm_ensemble.infer.domain.entities.llm_judgement import LLMJudgement
from llm_ensemble.aggregate.application.ports.driven.for_aggregating import ForAggregating
from llm_ensemble.aggregate.domain.entities.aggregated_vote import AggregatedVote
from llm_ensemble.aggregate.domain.entities.aggregation_strategy import AggregationStrategy
from llm_ensemble.aggregate.domain.aggregated_vote_builder import build_aggregated_vote
from llm_ensemble.libs.schemas import RelevanceScore


class MajorityVoteAdapter(ForAggregating):
    """Simple majority vote aggregation strategy adapter.

    Counts votes for each label and selects the label with the most votes.

    Tie handling: If multiple labels have the same count, picks the lowest
    numeric label deterministically (e.g., IRRELEVANT=0 beats RELEVANT=1).

    Confidence: Fraction of votes that went to the winning label.

    Strategy identity comes from config via constructor.
    """

    def __init__(self, strategy_name: str):
        """Initialize majority vote adapter with strategy name.

        Args:
            strategy_name: Natural key for AggregationStrategy entity (from config)
        """
        self.strategy_name = strategy_name

    def aggregate(self, judgements: list[LLMJudgement]) -> AggregatedVote:
        """Apply majority vote logic and create AggregatedVote entity.

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
            aggregation_strategy = self.get_strategy()
            return build_aggregated_vote(
                aggregation_strategy=aggregation_strategy,
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

        # Break ties deterministically: pick lowest numeric label
        final_label = min(winners, key=lambda x: x.value)

        # Calculate confidence: fraction of votes for winning label
        total_votes = len(valid_labels)
        final_confidence = max_count / total_votes

        # Build reasoning string
        if len(winners) > 1:
            reasoning = (
                f"{max_count}/{total_votes} models voted {final_label.label} "
                f"(tie with {', '.join(str(w.label) for w in winners if w != final_label)}, "
                f"broken by lowest label)"
            )
        else:
            reasoning = f"{max_count}/{total_votes} models voted {final_label.label}"

        # Build AggregatedVote entity
        aggregation_strategy = self.get_strategy()
        return build_aggregated_vote(
            aggregation_strategy=aggregation_strategy,
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
