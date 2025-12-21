"""Domain logic for inference metrics calculation."""

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
