"""Domain logic for inference metrics calculation."""

from llm_ensemble.infer.domain.entities.llm_judgement import LLMJudgement


def calculate_agreement(judgement: LLMJudgement) -> int:
    """Calculate whether LLM judgement agrees with gold score.
    
    Business rule: Agreement is binary (1 or 0) based on exact label match.
    
    Args:
        judgement: LLM judgement containing predicted and gold scores
        
    Returns:
        1 if extracted score matches gold score, 0 otherwise
    """
    extracted_score = judgement.llm_score.label.value if judgement.llm_score and judgement.llm_score.label else "null"
    gold_score = judgement.dataset_sample.judging_sample.gold_score.value
    return 1 if extracted_score == gold_score else 0
