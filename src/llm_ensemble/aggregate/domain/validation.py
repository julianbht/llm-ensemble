"""Domain validation functions for aggregate entities.

Pure domain logic for validating business rules and invariants.
"""

from __future__ import annotations
from uuid import UUID

from llm_ensemble.infer.domain.entities.llm_judgement import LLMJudgement


def validate_judgements_same_sample(
    judgements: list[LLMJudgement],
) -> None:
    """Validate that all judgements are for the same dataset_sample.

    This is a business rule invariant: when aggregating judgements, they must
    all be for the same query-document pair (dataset_sample).

    Args:
        judgements: List of LLM judgements to validate

    Raises:
        ValueError: If judgements is empty or judgements have different dataset_sample_ids
    """
    if not judgements:
        raise ValueError("llm_judgements cannot be empty")

    # Extract dataset_sample_id from first judgement
    dataset_sample_id: UUID = judgements[0].dataset_sample.id

    # Validate all judgements are for the same sample
    for judgement in judgements[1:]:
        if judgement.dataset_sample.id != dataset_sample_id:
            raise ValueError(
                f"All llm_judgements must be for the same dataset_sample. "
                f"Expected {dataset_sample_id}, got {judgement.dataset_sample.id}"
            )
