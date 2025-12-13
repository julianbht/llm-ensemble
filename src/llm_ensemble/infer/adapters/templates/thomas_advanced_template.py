"""Thomas et al. advanced prompt template.

Bundles the thomas-advanced prompt builder and parser together
to ensure they are always correctly paired.
"""

from __future__ import annotations

from llm_ensemble.infer.ports import PromptTemplatePort, PromptBuilderPort, ResponseParserPort
from llm_ensemble.infer.adapters.prompts.thomas_advanced_prompt_builder import ThomasAdvancedPromptBuilder
from llm_ensemble.infer.adapters.parsers.thomas_advanced_parser import ThomasAdvancedParser


class ThomasAdvancedTemplate(PromptTemplatePort):
    """Thomas et al. advanced prompt template.

    Bundles together:
    - ThomasAdvancedPromptBuilder: Renders prompts with role description and multi-aspect scoring
    - ThomasAdvancedParser: Parses {"M": N, "T": N, "O": N} JSON responses

    This ensures the prompt and parser are always correctly paired.
    """

    TEMPLATE_NAME = "thomas-advanced"

    def __init__(self):
        """Initialize template with builder and parser."""
        self._builder = ThomasAdvancedPromptBuilder()
        self._parser = ThomasAdvancedParser()

    def get_builder(self) -> PromptBuilderPort:
        """Get the prompt builder for this template.

        Returns:
            ThomasAdvancedPromptBuilder instance
        """
        return self._builder

    def get_parser(self) -> ResponseParserPort:
        """Get the response parser for this template.

        Returns:
            ThomasAdvancedParser instance
        """
        return self._parser

    def get_name(self) -> str:
        """Get the template name.

        Returns:
            'thomas-advanced'
        """
        return self.TEMPLATE_NAME
