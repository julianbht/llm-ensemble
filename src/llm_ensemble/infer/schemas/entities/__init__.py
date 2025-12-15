"""Domain entities for the infer CLI.

All persisted entities - pure domain models without infrastructure concerns.
"""

from llm_ensemble.infer.schemas.entities.llm_prompt import LLMPrompt
from llm_ensemble.infer.schemas.entities.llm_invocation_metrics import LLMInvocationMetrics
from llm_ensemble.infer.schemas.entities.llm_score import LLMScore
from llm_ensemble.infer.schemas.entities.llm_judgement import LLMJudgement

__all__ = [
    "LLMPrompt",
    "LLMInvocationMetrics",
    "LLMScore",
    "LLMJudgement",
]
