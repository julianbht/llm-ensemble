"""Prompt builder for LLM judge instructions.

Pure logic for building prompts from templates and JudgingExamples.
"""

from __future__ import annotations
from typing import Optional
from jinja2 import Template


def build_instruction(
    template: Template,
    query: str,
    page_text: str,
    role: bool = True,
    description: Optional[str] = None,
    narrative: Optional[str] = None,
    aspects: bool = False,
) -> str:
    """Build an LLM instruction from a Jinja2 template.

    Args:
        template: Jinja2 Template object (loaded by caller)
        query: The search query text
        page_text: The document/web page text to evaluate
        role: Whether to include the role description
        description: Optional query description
        narrative: Optional query narrative
        aspects: Whether to include multi-aspect evaluation (M/T/O)

    Returns:
        Rendered instruction string ready for LLM input

    Example:
        >>> from jinja2 import Template
        >>> template = Template("Query: {{ query }}")
        >>> build_instruction(template, "python tutorial", "Learn Python...")
        'Query: python tutorial'
    """
    return template.render(
        query=query,
        page_text=page_text,
        role=role,
        description=description,
        narrative=narrative,
        aspects=aspects,
    )


def build_instruction_from_judging_example(
    template: Template,
    example: dict,
    role: bool = True,
    description: Optional[str] = None,
    narrative: Optional[str] = None,
    aspects: bool = False,
) -> str:
    """Build instruction from a JudgingExample dict.

    Convenience wrapper that extracts query_text and doc from a JudgingExample.

    Args:
        template: Jinja2 Template object
        example: Dict with 'query_text' and 'doc' keys (JudgingExample schema)
        role: Whether to include the role description
        description: Optional query description
        narrative: Optional query narrative
        aspects: Whether to include multi-aspect evaluation

    Returns:
        Rendered instruction string

    Example:
        >>> example = {"query_text": "python", "doc": "Python is..."}
        >>> build_instruction_from_judging_example(template, example)
        'You are a search quality rater...'
    """
    return build_instruction(
        template=template,
        query=example["query_text"],
        page_text=example["doc"],
        role=role,
        description=description,
        narrative=narrative,
        aspects=aspects,
    )
