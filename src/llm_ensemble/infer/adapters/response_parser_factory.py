"""Factory for instantiating response parser adapters.

Creates concrete ResponseParser implementations based on explicit configuration.
Delegates to PromptConfig's built-in instantiation methods, making the config
the single source of truth for adapter selection.
"""

from __future__ import annotations

from llm_ensemble.infer.ports import ResponseParser
from llm_ensemble.infer.schemas import PromptConfig


def get_response_parser(prompt_config: PromptConfig) -> ResponseParser:
    """Instantiate a response parser adapter based on configuration.

    Factory function that delegates to PromptConfig's get_response_parser() method,
    which dynamically instantiates the parser from the module path.

    This factory follows explicit configuration principles - no implicit defaults.

    Args:
        prompt_config: Prompt configuration specifying parser module path

    Returns:
        Concrete ResponseParser implementation

    Raises:
        ImportError: If the parser module path cannot be imported

    Example:
        >>> from llm_ensemble.infer.config_loaders import load_prompt_config
        >>> config = load_prompt_config("thomas-et-al-prompt")
        >>> parser = get_response_parser(config)
        >>> score = parser.parse('{"O": 2}')
        >>> score.label
        2
    """
    # Use default score field "O" for now
    # TODO: Could be extended to support custom field names from config
    return prompt_config.get_response_parser(score_field="O")
