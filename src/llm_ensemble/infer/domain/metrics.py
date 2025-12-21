"""Domain logic for inference metrics calculation."""

from typing import Optional, TypedDict
from llm_ensemble.infer.domain.entities.llm_judgement import LLMJudgement


class AggregateStatistics(TypedDict):
    """Aggregate statistics calculated from a collection of judgements."""
    judgement_count: int
    error_count: int
    total_latency_ms: float
    avg_latency_ms: float
    warnings_summary: Optional[dict[str, int]]


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


def calculate_aggregate_statistics(judgements: list[LLMJudgement]) -> AggregateStatistics:
    """Calculate aggregate statistics from collection of judgements.
    
    Business rules:
    - Error = judgement with no score or no label
    - Average latency = total latency / count
    - Warnings aggregated by type
    
    Args:
        judgements: List of LLM judgements to aggregate
        
    Returns:
        AggregateStatistics with counts, latencies, and warnings summary
    """
    count = len(judgements)
    error_count = sum(1 for j in judgements if j.llm_score is None or j.llm_score.label is None)
    total_latency_ms = sum(j.llm_invocation_metrics.latency_ms for j in judgements)
    avg_latency_ms = total_latency_ms / count if count > 0 else 0.0
    
    # Aggregate warnings by type
    warnings_summary: dict[str, int] = {}
    for judgement in judgements:
        for warning in judgement.parser_warnings:
            warning_type = warning.__class__.__name__
            warnings_summary[warning_type] = warnings_summary.get(warning_type, 0) + 1
    
    return {
        "judgement_count": count,
        "error_count": error_count,
        "total_latency_ms": total_latency_ms,
        "avg_latency_ms": avg_latency_ms,
        "warnings_summary": warnings_summary if warnings_summary else None,
    }
