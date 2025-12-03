"""Majority vote aggregation strategy adapter.

Simple majority vote: the label with the most votes wins.
Ties are broken deterministically by choosing the lowest numeric label.
"""

from __future__ import annotations
from collections import Counter

from llm_ensemble.infer.schemas.llm_judgement import LLMJudgement
from llm_ensemble.aggregate.ports import AggregationStrategyPort
from llm_ensemble.aggregate.registry import AggregationStrategyRegistry
from llm_ensemble.libs.schemas import RelevanceScore


@AggregationStrategyRegistry.register("majority_vote")
class MajorityVoteAdapter(AggregationStrategyPort):
    """Simple majority vote aggregation strategy adapter.

    Counts votes for each label and selects the label with the most votes.

    Tie handling: If multiple labels have the same count, picks the lowest
    numeric label deterministically (e.g., IRRELEVANT=0 beats RELEVANT=1).

    Confidence: Fraction of votes that went to the winning label.

    Implements pure voting logic in aggregate_raw(), returning simple dict.
    Domain object creation handled by base class.
    Strategy identity comes from config via constructor.
    """

    def aggregate_raw(
        self,
        judgements: list[LLMJudgement]
    ) -> dict:
        """Apply majority vote logic - return raw vote data.

        Args:
            judgements: All model judgements for a single (query_id, docid) pair

        Returns:
            dict with final_label, final_confidence, final_reasoning
        """
        # Extract valid labels for voting
        valid_labels: list[RelevanceScore] = []

        for judgement in judgements:
            # Get label from llm_score (if available)
            if judgement.llm_score is not None and judgement.llm_score.label is not None:
                valid_labels.append(judgement.llm_score.label)

        # Handle edge case: no valid votes
        if not valid_labels:
            return {
                "final_label": None,
                "final_confidence": 0.0,
                "final_reasoning": "No valid votes (all models failed to parse)",
            }

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

        return {
            "final_label": final_label,
            "final_confidence": final_confidence,
            "final_reasoning": reasoning,
        }
