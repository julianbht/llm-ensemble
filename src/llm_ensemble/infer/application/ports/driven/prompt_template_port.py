"""Port interface for prompt templates.

Defines the abstract contract for prompt templates that bundle
a prompt builder and response parser together as a cohesive unit.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

from llm_ensemble.infer.application.ports.driven.prompt_builder_port import PromptBuilderPort
from llm_ensemble.infer.application.ports.driven.response_parser_port import ResponseParserPort


class PromptTemplatePort(ABC):
    """Abstract interface for prompt templates.

    A prompt template bundles a prompt builder and response parser that
    are designed to work together. This ensures that prompts and parsers
    are always correctly paired, preventing mismatches.

    The template is responsible for:
    1. Providing a prompt builder that renders the template
    2. Providing a response parser that understands the template's output format
    """

    @abstractmethod
    def get_builder(self) -> PromptBuilderPort:
        """Get the prompt builder for this template.

        Returns:
            PromptBuilderPort that renders this template's prompts
        """
        pass

    @abstractmethod
    def get_parser(self) -> ResponseParserPort:
        """Get the response parser for this template.

        Returns:
            ResponseParserPort that parses this template's responses
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the template name.

        Returns:
            Template name (e.g., 'thomas-simple')
        """
        pass
