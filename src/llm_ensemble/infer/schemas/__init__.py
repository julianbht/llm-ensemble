"""Pydantic schemas for the infer CLI.

Centralizes all data structure definitions.
"""

# NOTE: LLMResponse and LLMJudgement not imported here to avoid circular imports
# (they depend on JudgingSample which depends on IngestManifest which depends on IngestIOConfig which depends on IOConfig)
# Import them directly: from llm_ensemble.infer.schemas.llm_response import LLMResponse

from llm_ensemble.infer.schemas.model_config_schema import ModelConfig
from llm_ensemble.infer.schemas.prompt_config_schema import PromptConfig
from llm_ensemble.infer.schemas.io_config_schema import IOConfig
from llm_ensemble.infer.schemas.infer_manifest import InferManifest

__all__ = [
    "ModelConfig",
    "PromptConfig",
    "IOConfig",
    "InferManifest",
]
