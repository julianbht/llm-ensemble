"""Port interface for prompt builders.

Defines the abstract contract that all prompt builder adapters must implement.
This allows the system to work with different prompt formats and templates
without coupling to specific implementations.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

from llm_ensemble.ingest.schemas import JudgingExample


class PromptBuilder(ABC):
    """Abstract base class for prompt builders.

    Implementations can build prompts using different templates and formats
    while providing a consistent interface to the LLM provider adapters.

    Example:
        >>> class JinjaPromptBuilder(PromptBuilder):
        ...     def build(self, example):
        ...         return self.template.render(
        ...             query=example.query_text,
        ...             page_text=example.doc
        ...         )
    """

    @abstractmethod
    def build(self, example: JudgingExample) -> str:
        """Build a prompt from a judging example.

        Args:
            example: JudgingExample object containing query and document

        Returns:
            Rendered prompt string ready for LLM input

        Raises:
            ValueError: If example is invalid or missing required fields
        """
        pass
