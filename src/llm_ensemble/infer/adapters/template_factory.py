"""Factory for prompt template adapters.

Simple, explicit mapping of template names to template adapter classes.
No decorators, no hidden registration - just a clear dictionary.

To add a new template:
1. Create template adapter class that extends PromptTemplatePort
2. Import it here
3. Add to TEMPLATES dict
"""

from __future__ import annotations
from typing import Dict, Type

from llm_ensemble.infer.ports import PromptTemplatePort, PromptBuilderPort, ResponseParserPort
from llm_ensemble.infer.adapters.templates.thomas_simple_template import ThomasSimpleTemplate
from llm_ensemble.infer.adapters.templates.thomas_advanced_template import ThomasAdvancedTemplate


TEMPLATES: Dict[str, Type[PromptTemplatePort]] = {
    "thomas-simple": ThomasSimpleTemplate,
    "thomas-advanced": ThomasAdvancedTemplate,
}


class PromptTemplateFactory:
    """Factory for creating prompt template instances.

    Each template bundles a prompt builder and response parser that are
    designed to work together, ensuring they are always correctly paired.
    """

    @staticmethod
    def create(template_name: str) -> tuple[PromptBuilderPort, ResponseParserPort]:
        """Create and return prompt builder and parser for the template.

        Args:
            template_name: Name of the template (e.g., 'thomas-simple')

        Returns:
            Tuple of (prompt_builder, response_parser)

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
        template = template_class()
        return template.get_builder(), template.get_parser()

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
