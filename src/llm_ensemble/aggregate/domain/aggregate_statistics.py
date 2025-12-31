"""Domain functions for computing aggregate run statistics.

Pure domain logic for calculating metrics and building summaries from domain entities.
"""

from __future__ import annotations
from typing import Optional

from llm_ensemble.infer.domain.entities.infer_run_output import InferRunOutput
from llm_ensemble.aggregate.domain.entities.aggregated_vote import AggregatedVote


def calculate_aggregate_statistics(
    judged_datasets: list[InferRunOutput],
    aggregated_votes: list[AggregatedVote],
) -> tuple[int, int, int, int, Optional[dict[str, int]]]:
    """Calculate aggregate statistics from aggregation run.

    Business rules:
    - Total judgements = sum of all judgements across input datasets
    - Unique pairs = number of aggregated votes produced
    - Tie = aggregated vote with "tie" in reasoning
    - No valid votes = aggregated vote with final_label=None
    - Warnings aggregated by type

    Args:
        judged_datasets: List of InferRunOutput objects from aggregation inputs
        aggregated_votes: List of AggregatedVote entities produced

    Returns:
        Tuple of (total_judgements, unique_pairs, tie_count, no_valid_votes_count, warnings_summary)
    """
    # Calculate total judgements
    total_judgements = sum(
        len(judged_dataset.llm_judgements)
        for judged_dataset in judged_datasets
    )

    # Unique pairs is just the count of aggregated votes
    unique_pairs = len(aggregated_votes)

    # Count ties and no valid votes
    tie_count = 0
    no_valid_votes_count = 0

    for vote in aggregated_votes:
        if vote.final_label is None:
            no_valid_votes_count += 1
        elif vote.final_reasoning and "tie" in vote.final_reasoning.lower():
            tie_count += 1

    # Build warnings summary
    warnings_summary: dict[str, int] = {}
    if tie_count > 0:
        warnings_summary["tie"] = tie_count
    if no_valid_votes_count > 0:
        warnings_summary["no_valid_votes"] = no_valid_votes_count

    return (
        total_judgements,
        unique_pairs,
        tie_count,
        no_valid_votes_count,
        warnings_summary if warnings_summary else None,
    )
