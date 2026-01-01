"""Builder for EvaluationData entities with domain validation.

Domain layer - Business Rules

Validates evaluation data consistency before creating entities.
This ensures business rules are enforced at the domain boundary.
"""

from __future__ import annotations
from typing import Optional

from llm_ensemble.evaluate.domain.entities.evaluation_data import EvaluationData
from llm_ensemble.libs.schemas.relevance_score import RelevanceScore


def build_evaluation_data(
    ground_truth: list[RelevanceScore],
    predictions: list[Optional[RelevanceScore]],
    run_name: str,
    run_type: str,
) -> EvaluationData:
    """Build EvaluationData with domain validation.

    Business rules:
    - ground_truth and predictions must have same length
    - sample_count must match actual data length
    - All ground_truth values must be valid RelevanceScore

    Args:
        ground_truth: Ground truth relevance labels
        predictions: Predicted relevance labels (None if parse failed)
        run_name: Input run name
        run_type: Type of run (infer or aggregate)

    Returns:
        Validated EvaluationData entity

    Raises:
        ValueError: If business rules violated
    """
    # Business rule: ground_truth and predictions must have same length
    if len(ground_truth) != len(predictions):
        raise ValueError(
            f"Business rule violation: ground_truth and predictions must have same length. "
            f"Got {len(ground_truth)} ground truth labels vs {len(predictions)} predictions."
        )

    # Business rule: sample_count must match actual data
    sample_count = len(ground_truth)

    # Business rule: must have at least one sample
    if sample_count == 0:
        raise ValueError(
            "Business rule violation: cannot create EvaluationData with zero samples"
        )

    return EvaluationData(
        ground_truth=ground_truth,
        predictions=predictions,
        run_name=run_name,
        run_type=run_type,
        sample_count=sample_count,
    )
