"""Pydantic schemas for the infer CLI.

Centralizes all data structure definitions.
"""

from llm_ensemble.infer.schemas.model_judgement_schema import ModelJudgement
from llm_ensemble.infer.schemas.model_config_schema import ModelConfig
from llm_ensemble.infer.schemas.prompt_config_schema import PromptConfig

__all__ = [
    "ModelJudgement",
    "ModelConfig",
    "PromptConfig",
]
