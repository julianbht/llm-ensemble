"""Domain logic for inference metrics calculation.

Pure domain functions for calculating metrics from judgement entities.
Only contains functions with non-trivial business logic.
"""

from typing import Optional
from llm_ensemble.infer.domain.entities.llm_judgement import LLMJudgement


def get_extracted_score(judgement: LLMJudgement) -> Optional[int]:
    """Extract the predicted score value from judgement.

    Business rule: Return None if parsing failed.

    Args:
        judgement: LLM judgement containing predicted score

    Returns:
        Extracted score value (0-3) or None if unavailable
    """
    return judgement.llm_score.label.value if (judgement.llm_score and judgement.llm_score.label) else None


def calculate_latency_seconds(judgement: LLMJudgement) -> float:
    """Convert latency from milliseconds to seconds.

    Business rule: Latency is measured in milliseconds in metrics but reported in seconds.

    Args:
        judgement: LLM judgement containing invocation metrics

    Returns:
        Latency in seconds
    """
    return judgement.llm_invocation_metrics.latency_ms / 1000


def calculate_agreement(judgement: LLMJudgement) -> int:
    """Calculate whether LLM judgement agrees with gold score.

    Business rule: Agreement is binary (1 or 0) based on exact value match.

    Args:
        judgement: LLM judgement containing predicted and gold scores

    Returns:
        1 if extracted score matches gold score, 0 otherwise (including parse failures)
    """
    return 1 if (judgement.llm_score and judgement.llm_score.label and
                 judgement.llm_score.label.value == judgement.dataset_sample.judging_sample.gold_score.value) else 0


def count_errors(judgements: list[LLMJudgement]) -> int:
    """Count judgements with parsing errors.

    Business rule: Error when llm_score is None or label is None.

    Args:
        judgements: List of all LLM judgements produced

    Returns:
        Number of judgements with errors
    """
    return sum(
        1 for j in judgements
        if j.llm_score is None or j.llm_score.label is None
    )


def calculate_average_latency(judgements: list[LLMJudgement]) -> float:
    """Calculate average latency per judgement in milliseconds.

    Business rule: Total latency / count (returns 0.0 for empty list).

    Args:
        judgements: List of all LLM judgements produced

    Returns:
        Average latency in milliseconds
    """
    count = len(judgements)
    if count == 0:
        return 0.0
    total_latency_ms = sum(j.llm_invocation_metrics.latency_ms for j in judgements)
    return total_latency_ms / count


def aggregate_parser_warnings(judgements: list[LLMJudgement]) -> Optional[dict[str, int]]:
    """Aggregate parser warnings by type across all judgements.

    Business rule: Count occurrences of each warning type by class name.

    Args:
        judgements: List of all LLM judgements produced

    Returns:
        Dictionary mapping warning type to count, or None if no warnings
    """
    warnings_summary: dict[str, int] = {}
    for judgement in judgements:
        for warning in judgement.parser_warnings:
            warning_type = warning.__class__.__name__
            warnings_summary[warning_type] = warnings_summary.get(warning_type, 0) + 1

    return warnings_summary if warnings_summary else None
