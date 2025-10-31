"""Majority vote aggregation strategy adapter.

Simple majority vote: the label with the most votes wins.
Ties are broken deterministically by choosing the lowest numeric label.
"""

from __future__ import annotations
from collections import Counter

from llm_ensemble.infer.schemas.llm_judgement import LLMJudgement
from llm_ensemble.aggregate.schemas.aggregated_score import AggregatedScore, PerModelVote
from llm_ensemble.aggregate.ports import AggregationStrategy
from llm_ensemble.libs.schemas import RelevanceScore


class MajorityVoteAdapter(AggregationStrategy):
    """Simple majority vote aggregation strategy adapter.
    
    Counts votes for each label and selects the label with the most votes.
    
    Tie handling: If multiple labels have the same count, picks the lowest
    numeric label deterministically (e.g., IRRELEVANT=0 beats RELEVANT=1).
    
    Confidence: Fraction of votes that went to the winning label.
    """
    
    def aggregate(
        self,
        judgements: list[LLMJudgement]
    ) -> AggregatedScore:
        """Apply majority vote aggregation to a list of model judgements.
        
        Args:
            judgements: All model judgements for a single (query_id, docid) pair
            
        Returns:
            AggregatedScore with majority vote result and metadata
        """
        # Extract per-model votes (track model_id and label)
        per_model_votes: list[PerModelVote] = []
        valid_labels: list[RelevanceScore] = []
        
        for judgement in judgements:
            # Get model ID from run_info
            model_id = judgement.run_info.model_config_name
            
            # Get label and confidence from llm_score (if available)
            label = None
            confidence = None
            if judgement.llm_score is not None:
                label = judgement.llm_score.label
                confidence = judgement.llm_score.confidence
            
            # Record vote
            per_model_votes.append(PerModelVote(
                model_id=model_id,
                label=label,
                confidence=confidence,
            ))
            
            # Track valid labels for voting
            if label is not None:
                valid_labels.append(label)
        
        # Handle edge case: no valid votes
        if not valid_labels:
            return AggregatedScore(
                strategy="majority_vote",
                per_model_votes=per_model_votes,
                final_relevance_score=None,
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
        
        return AggregatedScore(
            strategy="majority_vote",
            per_model_votes=per_model_votes,
            final_relevance_score=final_label,
            final_confidence=final_confidence,
            final_reasoning=reasoning,
        )
