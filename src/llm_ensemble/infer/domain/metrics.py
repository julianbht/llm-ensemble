"""Domain logic for inference metrics calculation."""

from llm_ensemble.infer.domain.entities.llm_judgement import LLMJudgement


def get_extracted_score(judgement: LLMJudgement) -> str:
    """Extract the predicted score label from judgement.
    
    Business rule: Return "null" if parsing failed.
    
    Args:
        judgement: LLM judgement containing predicted score
        
    Returns:
        Extracted score label or "null" if unavailable
    """
    if judgement.llm_score and judgement.llm_score.label:
        return judgement.llm_score.label.value
    return "null"


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
    
    Business rule: Agreement is binary (1 or 0) based on exact label match.
    
    Args:
        judgement: LLM judgement containing predicted and gold scores
        
    Returns:
        1 if extracted score matches gold score, 0 otherwise
    """
    extracted_score = get_extracted_score(judgement)
    gold_score = judgement.dataset_sample.judging_sample.gold_score.value
    return 1 if extracted_score == gold_score else 0
