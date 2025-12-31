"""Domain functions for computing aggregate run statistics.

Pure domain logic for calculating metrics from domain entities.
Only contains functions with non-trivial business logic.
"""

from __future__ import annotations
from typing import Optional

from llm_ensemble.infer.domain.entities.infer_run_output import InferRunOutput
from llm_ensemble.aggregate.domain.entities.aggregated_vote import AggregatedVote


def count_total_judgements(judged_datasets: list[InferRunOutput]) -> int:
    """Calculate total number of LLM judgements across all input datasets.

    Business rule: Sum all judgements from all input InferRunOutput objects.

    Args:
        judged_datasets: List of InferRunOutput objects from aggregation inputs

    Returns:
        Total count of LLM judgements
    """
    return sum(
        len(judged_dataset.llm_judgements)
        for judged_dataset in judged_datasets
    )


def count_ties(aggregated_votes: list[AggregatedVote]) -> int:
    """Count aggregated votes that resulted in ties.

    Business rule: Tie when final_reasoning contains "tie" (case-insensitive).

    Args:
        aggregated_votes: List of AggregatedVote entities produced

    Returns:
        Number of ties
    """
    return sum(
        1 for vote in aggregated_votes
        if vote.final_label is not None
        and vote.final_reasoning
        and "tie" in vote.final_reasoning.lower()
    )


def count_no_valid_votes(aggregated_votes: list[AggregatedVote]) -> int:
    """Count aggregated votes with no valid input votes.

    Business rule: No valid votes when final_label is None.

    Args:
        aggregated_votes: List of AggregatedVote entities produced

    Returns:
        Number of votes with no valid inputs
    """
    return sum(
        1 for vote in aggregated_votes
        if vote.final_label is None
    )


def build_warnings_summary(
    tie_count: int,
    no_valid_votes_count: int,
) -> Optional[dict[str, int]]:
    """Build warnings summary dictionary from warning counts.

    Business rule: Only include warning types with non-zero counts.

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
