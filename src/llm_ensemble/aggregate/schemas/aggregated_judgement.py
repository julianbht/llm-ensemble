"""AggregatedJudgement schema - ensemble consensus for a query-document pair.

This is the canonical output schema for the aggregate CLI, representing the
combined judgements from multiple models for a single query-document pair.
"""

from __future__ import annotations
from pydantic import BaseModel, Field

from llm_ensemble.infer.schemas.llm_judgement import LLMJudgement
from llm_ensemble.aggregate.schemas.aggregated_score import AggregatedScore


class AggregatedJudgement(BaseModel):
    """Ensemble consensus combining multiple model judgements for a query-document pair.
    
    Contains:
    - judgements: All individual model judgements (with full provenance)
    - aggregated_scores: Results from applying each strategy (typically one strategy per run)
    
    This schema enables multiple aggregation strategies to be applied to the same
    input judgements, producing different consensus decisions in parallel.
    """
    
    judgements: list[LLMJudgement] = Field(
        ...,
        description=(
            "All individual model judgements for this query-document pair. "
            "Each contains full provenance: sample, request, response, score, run_info."
        )
    )
    
    aggregated_scores: list[AggregatedScore] = Field(
        ...,
        description=(
            "Results from applying aggregation strategies. "
            "Typically contains one entry per run, but can contain multiple "
            "if multiple strategies are configured."
        )
    )
    
    def get_primary_aggregated_score(self) -> AggregatedScore:
        """Get the primary aggregated score (first strategy result).
        
        Convenience method for accessing the main consensus decision when only
        one strategy is configured (common case).
        
        Returns:
            First aggregated score from the list
            
        Raises:
            IndexError: If no aggregated scores exist
        """
        return self.aggregated_scores[0]
