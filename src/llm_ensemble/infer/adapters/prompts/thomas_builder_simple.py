"""Jinja2-based prompt builder adapter.

Prompt builder that uses the Thomas et al. template for relevance judging.
The template is colocated with this adapter in the templates/ subdirectory.
"""

from __future__ import annotations
from jinja2 import Template

from llm_ensemble.ingest.schemas import JudgingSample
from llm_ensemble.infer.ports import PromptBuilder
from llm_ensemble.infer.schemas.llm_request import LLMRequest
from llm_ensemble.infer.schemas.warnings import PromptWarning, PromptWarningCode
from llm_ensemble.libs.runtime.path_manager import PathManager


class JinjaPromptBuilder(PromptBuilder):
    """Prompt builder using Jinja2 templates.

    Loads the thomas-et-al-prompt.jinja template from the templates/ directory.
    Passes JudgingSample Pydantic model attributes directly to the template:
    - {{ query.query_text }} - The query text
    - {{ query.external_id }} - The query ID
    - {{ document.doc_text }} - The document text
    - {{ document.external_id }} - The document ID

    If you need a different template or mapping, create a new prompt builder
    adapter with its own template.

    Example:
        >>> builder = JinjaPromptBuilder()
        >>> from llm_ensemble.ingest.schemas import Query, Document, JudgingSample
        >>> from llm_ensemble.libs.schemas import RelevanceScore
        >>> query = Query(external_id="q1", query_text="python")
        >>> doc = Document(external_id="d1", doc_text="Python is a programming language")
        >>> example = JudgingSample(
        ...     query=query,
        ...     document=doc,
        ...     gold_score=RelevanceScore.HIGHLY_RELEVANT,
        ...     run_info=...
        ... )
        >>> request = builder.build(example)
        >>> "Query: python" in request.prompt
        True
    """

    def __init__(self):
        """Initialize Jinja prompt builder.

        Loads the template from templates/thomas-et-al-prompt.jinja using PathManager.
        """
        # Load template using PathManager
        template_path = PathManager.get_prompt_templates_dir() / "thomas-et-al-prompt.jinja"

        if not template_path.exists():
            raise FileNotFoundError(
                f"Template not found: {template_path}\n"
                f"Expected template to be at: {template_path}"
            )

        with open(template_path, "r", encoding="utf-8") as f:
            self.template_text = f.read()
            self.template = Template(self.template_text)

    def build(self, example: JudgingSample) -> LLMRequest:
        """Build a prompt from a judging sample.

        Passes JudgingSample model attributes to the template:
        - query: Query Pydantic object (with .query_text, .external_id)
        - document: Document Pydantic object (with .doc_text, .external_id)

        Args:
            example: JudgingSample object containing query and document

        Returns:
            LLMRequest containing rendered prompt and any warnings
        """
        warnings = []

        # Pass JudgingSample Pydantic model attributes directly to template
        template_vars = {
            "query": example.query.query_text,
            "document": example.document.doc_text,
        }

        # Render template (catch rendering errors and convert to warnings)
        try:
            prompt = self.template.render(**template_vars)
        except Exception as e:
            # Template rendering failed - this is recoverable, return warning
            warnings.append(
                PromptWarning(
                    code=PromptWarningCode.RENDERING_ERROR,
                    message=f"Template rendering failed: {str(e)}",
                    metadata={"error_type": type(e).__name__}
                )
            )
            # Return empty prompt on render failure
            prompt = ""

        return LLMRequest(prompt=prompt, warnings=warnings)

    def get_template_text(self) -> str:
        """Get the raw Jinja template text.

        Returns:
            Raw template string (unrendered Jinja template)
        """
        return self.template_text
