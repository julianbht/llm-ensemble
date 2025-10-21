"""Jinja2-based prompt builder adapter.

Generic implementation that uses Jinja2 templates to build prompts.
Works with any template format by mapping JudgingExample fields to template variables.
"""

from __future__ import annotations
from jinja2 import Template

from llm_ensemble.ingest.schemas import JudgingExample
from llm_ensemble.infer.ports import PromptBuilder


class JinjaPromptBuilder(PromptBuilder):
    """Prompt builder using Jinja2 templates.

    This is a generic adapter that can work with any Jinja2 template.
    It maps JudgingExample fields to template variables based on the
    configured variable mapping.

    Example:
        >>> from jinja2 import Template
        >>> template = Template("Query: {{ query }}\\nDocument: {{ page_text }}")
        >>> builder = JinjaPromptBuilder(template)
        >>> example = JudgingExample(
        ...     dataset="test",
        ...     query_id="q1",
        ...     query_text="python",
        ...     docid="d1",
        ...     doc="Python is a programming language"
        ... )
        >>> prompt = builder.build(example)
        >>> "Query: python" in prompt
        True
    """

    def __init__(
        self,
        template: Template,
        variable_mapping: dict[str, str] | None = None,
    ):
        """Initialize Jinja prompt builder.

        Args:
            template: Jinja2 Template object to render
            variable_mapping: Optional mapping from template variables to JudgingExample fields.
                If not provided, uses default mapping:
                - query -> query_text
                - page_text -> doc
        """
        self.template = template
        self.variable_mapping = variable_mapping or {
            "query": "query_text",
            "page_text": "doc",
        }

    def build(self, example: JudgingExample) -> str:
        """Build a prompt from a judging example.

        Args:
            example: JudgingExample object containing query and document

        Returns:
            Rendered prompt string ready for LLM input

        Raises:
            ValueError: If example is missing required fields
        """
        # Convert example to dict
        example_dict = example.model_dump()

        # Build template variables from example using mapping
        template_vars = {}
        for template_var, example_field in self.variable_mapping.items():
            if example_field not in example_dict:
                raise ValueError(
                    f"Example missing required field '{example_field}' "
                    f"for template variable '{template_var}'"
                )
            template_vars[template_var] = example_dict[example_field]

        # Render and return
        return self.template.render(**template_vars)
