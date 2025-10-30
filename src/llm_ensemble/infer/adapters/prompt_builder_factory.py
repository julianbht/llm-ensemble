"""Factory for instantiating prompt builder adapters.

Creates concrete PromptBuilder implementations based on explicit configuration.
Delegates to PromptConfig's built-in instantiation methods, making the config
the single source of truth for adapter selection.
"""

from __future__ import annotations

from llm_ensemble.infer.ports import PromptBuilder
from llm_ensemble.infer.schemas import PromptConfig
from llm_ensemble.infer.config_loaders import load_prompt_template


def get_prompt_builder(prompt_config: PromptConfig) -> PromptBuilder:
    """Instantiate a prompt builder adapter based on configuration.

    Factory function that delegates to PromptConfig's get_prompt_builder() method,
    which dynamically instantiates the builder from the module path.

    This factory follows explicit configuration principles - no implicit defaults.
    Template location is determined by the template loader.

    Args:
        prompt_config: Prompt configuration specifying builder module path

    Returns:
        Concrete PromptBuilder implementation

    Raises:
        ImportError: If the builder module path cannot be imported

    Example:
        >>> from llm_ensemble.infer.config_loaders import load_prompt_config
        >>> config = load_prompt_config("thomas-et-al-prompt")
        >>> builder = get_prompt_builder(config)
        >>> example = JudgingSample(...)
        >>> prompt = builder.build(example)
    """
    # Load the template (loader determines location)
    template = load_prompt_template(prompt_config.prompt_template)

    # Delegate to config's instantiation method
    return prompt_config.get_prompt_builder(template)
