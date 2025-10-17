"""Pydantic schemas for the infer CLI.

Centralizes all data structure definitions.
"""

from llm_ensemble.infer.schemas.model_judgement import ModelJudgement
from llm_ensemble.infer.schemas.model_config import ModelConfig
from llm_ensemble.infer.schemas.prompt_config import PromptConfig

__all__ = [
    "ModelJudgement",
    "ModelConfig",
    "PromptConfig",
]
