"""Config feature for model and prompt configuration loading."""

from llm_ensemble.infer.configs.schemas import ModelConfig, PromptConfig
from llm_ensemble.infer.configs.models_loader import load_model_config, get_endpoint_url
from llm_ensemble.infer.configs.prompts_loader import load_prompt_config

__all__ = [
    "ModelConfig",
    "PromptConfig",
    "load_model_config",
    "get_endpoint_url",
    "load_prompt_config",
]
