"""Config feature for model and prompt configuration loading."""

from llm_ensemble.infer.config.models import ModelConfig, PromptConfig
from llm_ensemble.infer.config.loader import load_model_config, get_endpoint_url
from llm_ensemble.infer.config.prompts import load_prompt_config

__all__ = [
    "ModelConfig",
    "PromptConfig",
    "load_model_config",
    "get_endpoint_url",
    "load_prompt_config",
]
