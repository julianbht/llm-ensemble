"""Factory for instantiating response parser adapters.

Creates concrete ResponseParser implementations based on explicit configuration.
Follows the same pattern as io_factory and provider_factory.
"""

from __future__ import annotations

from llm_ensemble.infer.ports import ResponseParser
from llm_ensemble.infer.schemas import PromptConfig
from llm_ensemble.infer.adapters.parsers.json_response_parser import JsonResponseParser


def get_response_parser(prompt_config: PromptConfig) -> ResponseParser:
    """Instantiate a response parser adapter based on configuration.

    This factory follows explicit configuration principles - no implicit defaults.
    All parser types must be explicitly specified in the prompt config.

    Args:
        prompt_config: Prompt configuration specifying parser type

    Returns:
        Concrete ResponseParser implementation

    Raises:
        ValueError: If response_parser type is not recognized or missing

    Example:
        >>> from llm_ensemble.infer.config_loaders import load_prompt_config
        >>> config = load_prompt_config("thomas-et-al-prompt")
        >>> parser = get_response_parser(config)
        >>> label, warnings = parser.parse('{"O": 2}')
        >>> label
        2
    """
    parser_type = prompt_config.response_parser

    if not parser_type:
        raise ValueError(
            f"Prompt config '{prompt_config.name}' missing required field 'response_parser'. "
            f"Must explicitly specify parser type (e.g., 'json')."
        )

    # Map parser type to adapter implementation
    if parser_type == "json":
        # Use default score field "O" for now
        # TODO: Could be extended to support custom field names from config
        return JsonResponseParser(score_field="O")
    else:
        raise ValueError(
            f"Unknown response_parser type '{parser_type}' in prompt config '{prompt_config.name}'. "
            f"Supported types: 'json'"
        )
