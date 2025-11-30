"""Prompt-parser configuration loader.

Loads prompt-parser YAML configurations from the centralized configs/prompts directory.
"""

from __future__ import annotations

from llm_ensemble.infer.schemas.prompt_config_schema import PromptParserConfig
from llm_ensemble.libs.config import load_yaml_config
from llm_ensemble.libs.runtime.path_manager import PathManager


def load_prompt_parser_config(prompt_name: str) -> PromptParserConfig:
    """Load a prompt-parser configuration from YAML file.

    Args:
        prompt_name: Prompt identifier (e.g., "thomas-simple")

    Returns:
        PromptParserConfig object with all settings loaded from YAML

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If YAML is invalid or missing required fields
    """
    return load_yaml_config(
        config_name=prompt_name,
        config_dir=PathManager.get_prompts_dir(),
        schema=PromptParserConfig,
        config_type="prompt_parser_config",
    )
