"""Thomas et al. prompt builder.

Builds prompts for the Thomas et al. search quality rater format.
"""

from __future__ import annotations
from jinja2 import Template


def build(template: Template, example: dict) -> str:
    """Build a Thomas et al. prompt from template and judging example.

    Args:
        template: Jinja2 Template object loaded from configs/prompts/
        example: Dict with 'query_text' and 'doc' keys (JudgingExample schema)

    Returns:
        Rendered prompt string ready for LLM input

    Example:
        >>> from jinja2 import Template
        >>> template = Template("Query: {{ query }}")
        >>> example = {"query_text": "python", "doc": "Python is..."}
        >>> build(template, example)
        'Query: python'
    """
    # Map JudgingExample fields to template variables
    # Template expects: query, page_text
    # Example provides: query_text, doc
    return template.render(
        query=example["query_text"],
        page_text=example["doc"],
    )
