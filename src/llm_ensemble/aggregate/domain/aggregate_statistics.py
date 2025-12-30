"""Domain functions for computing aggregate run statistics.

Pure domain logic for calculating metrics and building summaries from domain entities.
"""

from __future__ import annotations
from typing import Optional

from llm_ensemble.infer.domain.entities.infer_run_output import InferRunOutput


def calculate_total_judgements(judged_datasets: list[InferRunOutput]) -> int:
    """Calculate total number of LLM judgements across all input datasets.

    Args:
        judged_datasets: List of InferRunOutput objects from aggregation inputs

    Returns:
        Total count of LLM judgements
    """
    return sum(
        len(judged_dataset.llm_judgements)
        for judged_dataset in judged_datasets
    )


def build_warnings_summary(
    tie_count: int,
    no_valid_votes_count: int,
) -> Optional[dict[str, int]]:
    """Build warnings summary from aggregation statistics.

    Business rules for what constitutes a warning:
    - Ties: When aggregation strategy couldn't reach consensus
    - No valid votes: When all judgements were invalid/null

    Args:
        tie_count: Number of samples with tied votes
        no_valid_votes_count: Number of samples with no valid votes

    Returns:
        Dictionary of warning types and counts, or None if no warnings
    """
    warnings_summary: dict[str, int] = {}

    if tie_count > 0:
        warnings_summary["tie"] = tie_count

    if no_valid_votes_count > 0:
        warnings_summary["no_valid_votes"] = no_valid_votes_count

    return warnings_summary if warnings_summary else None
