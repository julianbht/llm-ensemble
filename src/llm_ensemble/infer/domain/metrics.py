"""Domain logic for inference metrics calculation.

Pure domain functions for calculating metrics from judgement entities.
Only contains functions with non-trivial business logic.
"""

from typing import Optional
from llm_ensemble.infer.domain.entities.llm_judgement import LLMJudgement
from llm_ensemble.libs.schemas.relevance_score import RelevanceScore


def get_extracted_score(judgement: LLMJudgement) -> Optional[int]:
    """Extract the predicted score value from judgement.

    Business rule: Return None if parsing failed.

    Args:
        judgement: LLM judgement containing predicted score

    Returns:
        Extracted score value (0-3) or None if unavailable
    """
    if judgement.llm_score is None:
        return None

    if judgement.llm_score.label is None:
        return None

    return judgement.llm_score.label.value


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
    if judgement.llm_score is None:
        return 0

    if judgement.llm_score.label is None:
        return 0

    extracted = judgement.llm_score.label.value
    gold = judgement.dataset_sample.judging_sample.gold_score.value

    return 1 if extracted == gold else 0


def count_failed_parses(judgements: list[LLMJudgement]) -> int:
    """Count judgements where parsing failed.

    Business rule: Parsing failed when llm_score is None or label is None.

    Args:
        judgements: List of all LLM judgements produced

    Returns:
        Number of judgements where parsing failed
    """
    return sum(
        1 for j in judgements
        if j.llm_score is None or j.llm_score.label is None
    )


def calculate_vote_breakdown(judgements: list[LLMJudgement]) -> dict[str, int]:
    """Calculate count of judgements per relevance label.

    Business rule: Count how many judgements fall into each relevance score.
    Failed parses are tracked separately and not included in label counts.
    Uses RelevanceScore enum to ensure all possible labels are initialized.

    Args:
        judgements: List of all LLM judgements produced

    Returns:
        Dictionary mapping label name to count (e.g., {"Irrelevant": 120, "Relevant": 45, ...})
    """
    # Initialize counts for all possible relevance scores from enum
    vote_counts: dict[str, int] = {
        score.label: 0
        for score in RelevanceScore
    }

    # Count occurrences of each label
    for judgement in judgements:
        if judgement.llm_score and judgement.llm_score.label is not None:
            label_name = judgement.llm_score.label.label
            vote_counts[label_name] += 1

    return vote_counts


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


def aggregate_parse_issues(judgements: list[LLMJudgement]) -> Optional[dict[str, int]]:
    """Aggregate parse issues by code across all judgements.

    Business rule: Count occurrences of each issue code.

    Args:
        judgements: List of all LLM judgements produced

    Returns:
        Dictionary mapping issue code to count, or None if no issues
    """
    issues_summary: dict[str, int] = {}
    for judgement in judgements:
        if judgement.parser_issue:
            code = judgement.parser_issue.code.value
            issues_summary[code] = issues_summary.get(code, 0) + 1

    return issues_summary if issues_summary else None


def calculate_total_cost(judgements: list[LLMJudgement]) -> Optional[float]:
    """Calculate total estimated cost across all judgements.

    Business rule: Sum cost estimates where available, return None if no costs tracked.

    Args:
        judgements: List of all LLM judgements produced

    Returns:
        Total cost in USD, or None if no cost data available
    """
    total = sum(
        j.llm_invocation_metrics.cost_estimate_usd
        for j in judgements
        if j.llm_invocation_metrics.cost_estimate_usd is not None
    )
    return total if total > 0 else None


def calculate_total_prompt_tokens(judgements: list[LLMJudgement]) -> Optional[int]:
    """Calculate total prompt tokens across all judgements.

    Business rule: Sum prompt tokens where available, return None if no tokens tracked.

    Args:
        judgements: List of all LLM judgements produced

    Returns:
        Total prompt tokens, or None if no token data available
    """
    total = sum(
        j.llm_invocation_metrics.prompt_tokens
        for j in judgements
        if j.llm_invocation_metrics.prompt_tokens is not None
    )
    return total if total > 0 else None


def calculate_total_completion_tokens(judgements: list[LLMJudgement]) -> Optional[int]:
    """Calculate total completion tokens across all judgements.

    Business rule: Sum completion tokens where available, return None if no tokens tracked.

    Args:
        judgements: List of all LLM judgements produced

    Returns:
        Total completion tokens, or None if no token data available
    """
    total = sum(
        j.llm_invocation_metrics.completion_tokens
        for j in judgements
        if j.llm_invocation_metrics.completion_tokens is not None
    )
    return total if total > 0 else None


def calculate_total_tokens(judgements: list[LLMJudgement]) -> Optional[int]:
    """Calculate total tokens across all judgements.

    Business rule: Sum total tokens where available, return None if no tokens tracked.

    Args:
        judgements: List of all LLM judgements produced

    Returns:
        Total tokens (prompt + completion), or None if no token data available
    """
    total = sum(
        j.llm_invocation_metrics.total_tokens
        for j in judgements
        if j.llm_invocation_metrics.total_tokens is not None
    )
    return total if total > 0 else None


def calculate_eta_seconds(
    samples_processed: int,
    total_samples: int,
    elapsed_seconds: float,
) -> float:
    """Calculate estimated time remaining for processing samples.

    Business rule: ETA is based on average time per sample multiplied by remaining samples.
    If no samples processed yet, returns 0.0.

    Args:
        samples_processed: Number of samples completed so far
        total_samples: Total number of samples to process
        elapsed_seconds: Time elapsed since processing started

    Returns:
        Estimated seconds remaining to complete all samples
    """
    if samples_processed == 0:
        return 0.0

    avg_time_per_sample = elapsed_seconds / samples_processed
    remaining_samples = total_samples - samples_processed
    return avg_time_per_sample * remaining_samples


def format_eta(seconds: float) -> str:
    """Format ETA seconds into human-readable string.

    Business rule: Display hours, minutes, and seconds for readability.
    Format: "Xh Ym Zs" (omit zero components except when total is 0s).

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string (e.g., "2h 15m 30s", "45m 12s", "5s", "0s")
    """
    if seconds < 0:
        seconds = 0

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:  # Always show seconds if no other units
        parts.append(f"{secs}s")

    return " ".join(parts)
