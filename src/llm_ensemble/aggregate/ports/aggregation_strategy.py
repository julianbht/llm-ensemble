"""Port interface for aggregation strategies.

Defines the abstract contract that all aggregation strategy adapters must implement.
This allows the system to use different aggregation methods (majority vote, weighted,
etc.) without coupling to specific implementations.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

from llm_ensemble.infer.schemas.llm_judgement import LLMJudgement
from llm_ensemble.aggregate.schemas.aggregated_score import AggregatedScore


class AggregationStrategy(ABC):
    """Abstract base class for aggregation strategies.
    
    Implementations can use different aggregation methods (majority vote,
    weighted vote, soft voting, etc.) while providing a consistent interface
    to combine multiple model judgements into a consensus decision.
    """
    
    @abstractmethod
    def aggregate(self, judgements: list[LLMJudgement]) -> AggregatedScore:
        """Apply aggregation strategy to combine multiple model judgements.
        
        Args:
            judgements: All model judgements for a single (query_id, docid) pair
            
        Returns:
            AggregatedScore with consensus decision, confidence, and reasoning
            
        Note:
            Strategy should handle edge cases gracefully:
            - Empty judgements list
            - All models failed to parse (no valid labels)
            - Ties in voting
        """
        pass
