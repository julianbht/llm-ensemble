"""Factory for instantiating prompt builder adapters.

Creates concrete PromptBuilder implementations based on explicit configuration.
Follows the same pattern as io_factory and provider_factory.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

from llm_ensemble.infer.ports import PromptBuilder
from llm_ensemble.infer.schemas import PromptConfig
from llm_ensemble.infer.config_loaders import load_prompt_template
from llm_ensemble.infer.adapters.prompts.jinja_prompt_builder import JinjaPromptBuilder


def get_prompt_builder(
    prompt_config: PromptConfig,
    prompts_dir: Optional[Path] = None,
) -> PromptBuilder:
    """Instantiate a prompt builder adapter based on configuration.

    This factory follows explicit configuration principles - no implicit defaults.
    All builder types must be explicitly specified in the prompt config.

    Args:
        prompt_config: Prompt configuration specifying builder type
        prompts_dir: Directory containing prompt templates (defaults to configs/prompts)

    Returns:
        Concrete PromptBuilder implementation

    Raises:
        ValueError: If prompt_builder type is not recognized or missing

    Example:
        >>> from llm_ensemble.infer.config_loaders import load_prompt_config
        >>> config = load_prompt_config("thomas-et-al-prompt")
        >>> builder = get_prompt_builder(config)
        >>> example = JudgingExample(...)
        >>> prompt = builder.build(example)
    """
    builder_type = prompt_config.prompt_builder

    if not builder_type:
        raise ValueError(
            f"Prompt config '{prompt_config.name}' missing required field 'prompt_builder'. "
            f"Must explicitly specify builder type (e.g., 'jinja')."
        )

    # Load the template
    template = load_prompt_template(prompt_config.prompt_template, prompts_dir)

    # Map builder type to adapter implementation
    if builder_type == "jinja":
        # Use default variable mapping for now
        # TODO: Could be extended to support custom mappings from config
        return JinjaPromptBuilder(template)
    else:
        raise ValueError(
            f"Unknown prompt_builder type '{builder_type}' in prompt config '{prompt_config.name}'. "
            f"Supported types: 'jinja'"
        )
