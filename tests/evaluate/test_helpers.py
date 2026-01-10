"""Test helpers for evaluate tests.

Utilities for loading LLM Judge Challenge data and expected metrics.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

from llm_ensemble.libs.schemas.relevance_score import RelevanceScore


# Path to test fixtures
FIXTURES_DIR = Path(__file__).parent.parent
SCORES_DIR = FIXTURES_DIR / "scores"
QRELS_PATH = (
    FIXTURES_DIR.parent
    / "datasets"
    / "llm_judge_challenge_experiment"
    / "llm4eval_test_qrel_2024_recovered.txt"
)
EXPECTED_METRICS_PATH = FIXTURES_DIR / "llm-judge-challenge-metrics"


def load_trec_file(file_path: Path) -> list[RelevanceScore]:
    """Load TREC format file and return list of RelevanceScore labels.

    TREC format: query_id iteration doc_id relevance_score
    Example: q49 0 p3659 3

    Note: Some submissions contain invalid scores (e.g., 5). These are capped
    to the maximum valid score (3 = PERFECTLY_RELEVANT).

    Args:
        file_path: Path to TREC format file

    Returns:
        List of RelevanceScore labels in order
    """
    scores = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            relevance_score = int(parts[3])
            # Handle invalid scores by capping to max valid score (3)
            if relevance_score > 3:
                relevance_score = 3
            scores.append(RelevanceScore(relevance_score))
    return scores


def load_qrels() -> list[RelevanceScore]:
    """Load ground truth qrels file.

    Returns:
        List of ground truth RelevanceScore labels
    """
    return load_trec_file(QRELS_PATH)


def load_submission(submission_name: str) -> list[RelevanceScore]:
    """Load submission file by name.

    Args:
        submission_name: Name of submission (e.g., "TREMA-direct.txt")

    Returns:
        List of predicted RelevanceScore labels
    """
    submission_path = SCORES_DIR / submission_name
    return load_trec_file(submission_path)


def load_expected_metrics() -> dict[str, dict[str, float]]:
    """Load expected metrics from official results.

    Returns:
        Dict mapping submission_id to dict of metric_name -> value
        Example: {
            "TREMA-direct": {
                "cohenskappa": 0.1742,
                "krippendorfalpha": 0.3729,
                ...
            }
        }
    """
    metrics = {}
    with open(EXPECTED_METRICS_PATH, "r") as f:
        # Parse header
        header = f.readline().strip().split()
        metric_names = header[1:]  # Skip "submission-id" column

        # Parse rows
        for line in f:
            parts = line.strip().split()
            if len(parts) == 0:
                continue
            submission_id = parts[0]
            values = [float(v) for v in parts[1:]]
            metrics[submission_id] = dict(zip(metric_names, values))

    return metrics


def get_expected_metric(submission_id: str, metric_name: str) -> float:
    """Get expected metric value for a submission.

    Args:
        submission_id: Submission ID (without .txt extension)
        metric_name: Metric name (e.g., "cohenskappa", "krippendorfalpha")

    Returns:
        Expected metric value
    """
    all_metrics = load_expected_metrics()
    return all_metrics[submission_id][metric_name]
