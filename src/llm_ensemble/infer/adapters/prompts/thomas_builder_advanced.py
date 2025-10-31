"""Thomas et al. advanced prompt builder.

Prompt builder that uses the Thomas et al. template with advanced features enabled.
Enables role description and aspects-based evaluation in the prompt.
"""

from __future__ import annotations
from jinja2 import Template

from llm_ensemble.ingest.schemas import JudgingSample
from llm_ensemble.infer.ports import PromptBuilder
from llm_ensemble.infer.schemas.llm_request import LLMRequest
from llm_ensemble.infer.schemas.warnings import PromptWarning, PromptWarningCode
from llm_ensemble.libs.runtime.path_manager import PathManager


class ThomasBuilderAdvanced(PromptBuilder):
    """Advanced Thomas et al. prompt builder with role and aspects enabled.

    Loads the thomas-et-al-prompt.jinja template and enables advanced features:
    - role: True (shows search quality rater role description)
    - aspects: True (enables aspects-based evaluation with M, T, O scores)

    Passes query and document text directly to the template.

    Example:
        >>> builder = ThomasBuilderAdvanced()
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
        >>> "You are a search quality rater" in request.prompt
        True
    """

    def __init__(self):
        """Initialize advanced Thomas et al. prompt builder.

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
            self.template = Template(f.read())

    def build(self, example: JudgingSample) -> LLMRequest:
        """Build a prompt from a judging sample with advanced features enabled.

        Passes query and document text along with feature flags:
        - query: The query text string
        - document: The document text string
        - role: True (enables role description)
        - aspects: True (enables M, T, O scoring)

        Args:
            example: JudgingSample object containing query and document

        Returns:
            LLMRequest containing rendered prompt and any warnings
        """
        warnings = []

        # Pass query/document text and enable advanced features
        template_vars = {
            "query": example.query.query_text,
            "document": example.document.doc_text,
            "role": True,  # Enable role description
            "aspects": True,  # Enable aspects-based evaluation (M, T, O)
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
