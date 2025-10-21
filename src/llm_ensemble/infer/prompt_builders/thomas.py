"""Thomas et al. prompt builder adapter.

Builds prompts for the Thomas et al. search quality rater format.
Implements the PromptBuilder port interface.
"""

from __future__ import annotations
from jinja2 import Template

from llm_ensemble.infer.ports import PromptBuilder


class ThomasPromptBuilder(PromptBuilder):
    """Thomas et al. prompt builder implementation.

    Maps JudgingExample fields to template variables expected by the
    Thomas et al. prompt format.

    Example:
        >>> from jinja2 import Template
        >>> builder = ThomasPromptBuilder()
        >>> template = Template("Query: {{ query }}")
        >>> example = {"query_text": "python", "doc": "Python is..."}
        >>> builder.build(template, example)
        'Query: python'
    """

    def build(self, template: Template, example: dict) -> str:
        """Build a Thomas et al. prompt from template and judging example.

        Args:
            template: Jinja2 Template object loaded from configs/prompts/
            example: Dict with 'query_text' and 'doc' keys (JudgingExample schema)

        Returns:
            Rendered prompt string ready for LLM input

        Raises:
            ValueError: If required fields are missing from example
        """
        # Validate required fields
        if "query_text" not in example:
            raise ValueError("Example missing required field: query_text")
        if "doc" not in example:
            raise ValueError("Example missing required field: doc")

        # Map JudgingExample fields to template variables
        # Template expects: query, page_text
        # Example provides: query_text, doc
        return template.render(
            query=example["query_text"],
            page_text=example["doc"],
        )
