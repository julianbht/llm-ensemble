"""Thomas et al. simple prompt template.

Bundles the thomas-simple prompt builder and parser together
to ensure they are always correctly paired.
"""

from __future__ import annotations

from llm_ensemble.infer.ports.prompt_template_port import PromptTemplatePort
from llm_ensemble.infer.ports.prompt_builder_port import PromptBuilderPort
from llm_ensemble.infer.ports.response_parser_port import ResponseParserPort
from llm_ensemble.infer.adapters.prompts.thomas_simple_prompt_builder import ThomasSimplePromptBuilder
from llm_ensemble.infer.adapters.parsers.thomas_simple_parser import ThomasSimpleParser


class ThomasSimpleTemplate(PromptTemplatePort):
    """Thomas et al. simple prompt template.

    Bundles together:
    - ThomasSimplePromptBuilder: Renders prompts with simple 0-2 scoring instructions
    - ThomasSimpleParser: Parses {"O": N} JSON responses

    This ensures the prompt and parser are always correctly paired.
    """

    # Class-level constants for metadata access without instantiation
    TEMPLATE_NAME = "thomas-simple"
    TEMPLATE_TEXT = ThomasSimplePromptBuilder.TEMPLATE_TEXT
    TEMPLATE_ID = ThomasSimplePromptBuilder.TEMPLATE_ID
    BUILDER_NAME = ThomasSimplePromptBuilder.TEMPLATE_NAME
    PARSER_NAME = ThomasSimpleParser.PARSER_NAME

    def __init__(self):
        """Initialize template with builder and parser."""
        self._builder = ThomasSimplePromptBuilder()
        self._parser = ThomasSimpleParser()

    def get_builder(self) -> PromptBuilderPort:
        """Get the prompt builder for this template.

        Returns:
            ThomasSimplePromptBuilder instance
        """
        return self._builder

    def get_parser(self) -> ResponseParserPort:
        """Get the response parser for this template.

        Returns:
            ThomasSimpleParser instance
        """
        return self._parser

    def get_name(self) -> str:
        """Get the template name.

        Returns:
            'thomas-simple'
        """
        return self.TEMPLATE_NAME
