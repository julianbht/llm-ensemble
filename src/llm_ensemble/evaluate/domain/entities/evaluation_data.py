"""EvaluationData - normalized evaluation data entity.

Domain entity representing normalized evaluation data from either
infer or aggregate runs. This is the output of input adapters.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field

from llm_ensemble.libs.schemas.relevance_score import RelevanceScore


class EvaluationData(BaseModel):
    """Normalized evaluation data for metric computation.

    Input adapters read from infer/aggregate runs and normalize to this format.
    Metric adapters consume this entity to compute evaluation metrics.

    Use build_evaluation_data() from evaluation_data_builder.py to create instances.
    """

    ground_truth: list[RelevanceScore] = Field(
        ...,
        description="Ground truth relevance labels from dataset"
    )

    predictions: list[Optional[RelevanceScore]] = Field(
        ...,
        description="Predicted relevance labels from LLM or ensemble (None if parse failed)"
    )

    run_name: str = Field(
        ...,
        description="Input run name that was evaluated"
    )

    run_type: str = Field(
        ...,
        description="Type of run evaluated (infer or aggregate)"
    )

    sample_count: int = Field(
        ...,
        ge=0,
        description="Number of samples evaluated"
    )
