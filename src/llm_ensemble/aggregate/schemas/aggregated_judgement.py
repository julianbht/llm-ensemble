"""AggregatedJudgement schema - ensemble consensus for a query-document pair.

This is the canonical output schema for the aggregate CLI, representing the
combined judgements from multiple models for a single query-document pair.

Note: With the simplified schema, each run uses exactly one strategy, so this
schema now contains a single aggregated_score instead of a list.
"""

from __future__ import annotations
from pydantic import BaseModel, Field

from llm_ensemble.infer.schemas.llm_judgement import LLMJudgement
from llm_ensemble.aggregate.schemas.aggregated_score import AggregatedScore


class AggregatedJudgement(BaseModel):
    """Ensemble consensus combining multiple model judgements for a query-document pair.

    Contains:
    - judgements: All individual model judgements (with full provenance)
    - aggregated_score: Result from applying the run's aggregation strategy

    Each aggregate run uses exactly one strategy, so there is one score per judgement.
    To compare strategies, run the aggregate CLI multiple times with different strategies.
    """

    judgements: list[LLMJudgement] = Field(
        ...,
        description=(
            "All individual model judgements for this query-document pair."
        )
    )

    aggregated_score: AggregatedScore = Field(
        ...,
        description=(
            "Result from applying the aggregation strategy configured for this run."
        )
    )
