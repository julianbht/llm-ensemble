"""Port interface for prompt builders.

Defines the abstract contract that all prompt builder adapters must implement.
This allows the system to work with different prompt formats and templates
without coupling to specific implementations.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

from llm_ensemble.ingest.schemas import JudgingSample


class PromptBuilder(ABC):
    """Abstract base class for prompt builders.

    Implementations can build prompts using different templates and formats
    while providing a consistent interface to the LLM provider adapters.

    Returns the rendered prompt string directly (no wrapper needed).

    Example:
        >>> class JinjaPromptBuilder(PromptBuilder):
        ...     def build(self, example):
        ...         return self.template.render(
        ...             query=example.query.query_text,
        ...             document=example.document.doc_text
        ...         )
    """

    @abstractmethod
    def build(self, example: JudgingSample) -> str:
        """Build a prompt from a judging example.

        Args:
            example: JudgingSample object containing query and document

        Returns:
            Rendered prompt string

        Raises:
            Exception: For unrecoverable errors (e.g., template file missing).
        """
        pass

    @abstractmethod
    def get_template_text(self) -> str:
        """Get the raw template text for database storage.

        Returns the unrendered template string (e.g., Jinja template source).
        This enables storing templates in the database for analysis and filtering.

        Returns:
            Raw template text (before variable substitution)

        Example:
            >>> builder = JinjaPromptBuilder()
            >>> builder.get_template_text()
            'Query: {{ query }}\\nDocument: {{ document }}'
        """
        pass
