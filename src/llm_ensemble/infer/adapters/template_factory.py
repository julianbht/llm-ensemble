"""Factory for prompt template entities.

Simple, explicit mapping of template names to template adapter classes.
No decorators, no hidden registration - just a clear dictionary.

This factory returns PromptTemplate entities (pure metadata).
Adapter instantiation happens in the adapter_factory layer.

To add a new template:
1. Create template adapter class that extends PromptTemplatePort
2. Import it here
3. Add to TEMPLATES dict
"""

from __future__ import annotations
from typing import Dict, Type

from llm_ensemble.infer.ports.prompt_template_port import PromptTemplatePort
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
    """Factory for creating prompt template entities.

    Returns PromptTemplate entities with metadata only.
    Does NOT instantiate adapters - that's done in adapter_factory.
    """

    @staticmethod
    def create(template_name: str):
        """Get PromptTemplate entity with metadata.

        Creates metadata entities using class-level constants from template classes,
        avoiding the need to instantiate heavy adapter objects at config time.

        Args:
            template_name: Name of the template (e.g., 'thomas-simple')

        Returns:
            PromptTemplate entity with metadata (template_text, builder, parser names)

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
    def get_adapter_class(template_name: str) -> Type[PromptTemplatePort]:
        """Get template adapter class for instantiation.

        Used by adapter_factory to instantiate concrete adapters.

        Args:
            template_name: Name of the template (e.g., 'thomas-simple')

        Returns:
            Template adapter class

        Raises:
            ValueError: If template not found
        """
        if template_name not in TEMPLATES:
            available = ", ".join(sorted(TEMPLATES.keys()))
            raise ValueError(
                f"Template '{template_name}' not found. "
                f"Available: {available}"
            )

        return TEMPLATES[template_name]

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
