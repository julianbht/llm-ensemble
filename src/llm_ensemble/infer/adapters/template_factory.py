"""Factory for prompt templates.

Simple, explicit mapping of template names to template adapter classes.
No decorators, no hidden registration - just a clear dictionary.

To add a new template:
1. Create template adapter class that extends PromptTemplatePort
2. Import it here
3. Add to TEMPLATES dict
"""

from __future__ import annotations
from typing import Dict, Type

from llm_ensemble.infer.application.ports.driven.prompt_template_port import PromptTemplatePort
from llm_ensemble.infer.application.ports.driven.prompt_builder_port import PromptBuilderPort
from llm_ensemble.infer.application.ports.driven.response_parser_port import ResponseParserPort
from llm_ensemble.infer.adapters.templates.thomas_simple_template import ThomasSimpleTemplate
from llm_ensemble.infer.adapters.templates.thomas_advanced_template import ThomasAdvancedTemplate
from llm_ensemble.infer.domain.entities.prompt_template import PromptTemplate
from llm_ensemble.infer.domain.entities.prompt_builder import PromptBuilder
from llm_ensemble.infer.domain.entities.parser import Parser


TEMPLATES: Dict[str, Type[PromptTemplatePort]] = {
    "thomas-simple": ThomasSimpleTemplate,
    "thomas-advanced": ThomasAdvancedTemplate,
}


class PromptTemplateFactory:
    """Factory for creating prompt builders and parsers."""

    @staticmethod
    def create_builder(template_name: str) -> PromptBuilderPort:
        """Create prompt builder for the given template.

        Args:
            template_name: Name of the template (e.g., 'thomas-simple')

        Returns:
            PromptBuilderPort instance

        Raises:
            ValueError: If template not found
        """
        if template_name not in TEMPLATES:
            available = ", ".join(sorted(TEMPLATES.keys()))
            raise ValueError(
                f"Template '{template_name}' not found. "
                f"Available: {available}"
            )

        template_class = TEMPLATES[template_name]
        template_adapter = template_class()
        return template_adapter.get_builder()

    @staticmethod
    def create_parser(template_name: str) -> ResponseParserPort:
        """Create response parser for the given template.

        Args:
            template_name: Name of the template (e.g., 'thomas-simple')

        Returns:
            ResponseParserPort instance

        Raises:
            ValueError: If template not found
        """
        if template_name not in TEMPLATES:
            available = ", ".join(sorted(TEMPLATES.keys()))
            raise ValueError(
                f"Template '{template_name}' not found. "
                f"Available: {available}"
            )

        template_class = TEMPLATES[template_name]
        template_adapter = template_class()
        return template_adapter.get_parser()

    @staticmethod
    def create(template_name: str) -> PromptTemplate:
        """Get PromptTemplate entity with metadata for manifest persistence.

        Args:
            template_name: Name of the template (e.g., 'thomas-simple')

        Returns:
            PromptTemplate entity with metadata

        Raises:
            ValueError: If template not found
        """
        if template_name not in TEMPLATES:
            available = ", ".join(sorted(TEMPLATES.keys()))
            raise ValueError(
                f"Template '{template_name}' not found. "
                f"Available: {available}"
            )

        template_class = TEMPLATES[template_name]

        return PromptTemplate(
            name=template_class.TEMPLATE_NAME,
            template_text=template_class.TEMPLATE_TEXT,
            prompt_builder=PromptBuilder(name=template_class.BUILDER_NAME),
            response_text_parser=Parser(name=template_class.PARSER_NAME),
        )

    @staticmethod
    def list_available() -> list[str]:
        """List all available template names.

        Returns:
            Sorted list of template names
        """
        return sorted(TEMPLATES.keys())

    @staticmethod
    def has_template(template_name: str) -> bool:
        """Check if template is available.

        Args:
            template_name: Name of the template

        Returns:
            True if template exists
        """
        return template_name in TEMPLATES

    @staticmethod
    def get_description(template_name: str) -> str:
        """Get description for a template.

        Args:
            template_name: Name of the template

        Returns:
            Description string from template's docstring

        Raises:
            ValueError: If template not found
        """
        if template_name not in TEMPLATES:
            available = ", ".join(sorted(TEMPLATES.keys()))
            raise ValueError(
                f"Template '{template_name}' not found. "
                f"Available: {available}"
            )

        template_class = TEMPLATES[template_name]
        return template_class.__doc__.strip().split('\n')[0] if template_class.__doc__ else template_name
