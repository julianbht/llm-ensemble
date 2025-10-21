"""Port interface for prompt builders.

Defines the abstract contract that all prompt builder adapters must implement.
This allows the domain/providers to depend on an abstraction rather than concrete
builder implementations.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from jinja2 import Template


class PromptBuilder(ABC):
    """Abstract base class for prompt builders.

    All prompt builder adapters must inherit from this class and implement
    the build() method.

    Example:
        >>> class ThomasPromptBuilder(PromptBuilder):
        ...     def build(self, template, example):
        ...         return template.render(query=example["query_text"])
    """

    @abstractmethod
    def build(self, template: Template, example: dict) -> str:
        """Build a prompt from template and judging example.

        Args:
            template: Jinja2 Template object loaded from configs/prompts/
            example: Dict with judging example data (JudgingExample schema)

        Returns:
            Rendered prompt string ready for LLM input

        Raises:
            ValueError: If required fields are missing from example
            Exception: If template rendering fails
        """
        pass
