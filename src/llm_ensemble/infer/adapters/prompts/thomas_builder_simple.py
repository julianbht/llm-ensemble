"""Jinja2-based prompt builder adapter.

Prompt builder that uses the Thomas et al. template for relevance judging.
The template is colocated with this adapter in the templates/ subdirectory.
"""

from __future__ import annotations
from jinja2 import Template

from llm_ensemble.ingest.schemas.dataset_sample import DatasetSample
from llm_ensemble.infer.ports import PromptBuilder
from llm_ensemble.infer.schemas.llm_judgement import LLMPrompt
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
    """

    def __init__(self):
        """Initialize Jinja prompt builder.

        Loads the template from templates/thomas-simple.jinja using PathManager.
        """
        # Load template using PathManager
        template_path = PathManager.get_prompt_templates_dir() / "thomas-simple.jinja"

        if not template_path.exists():
            raise FileNotFoundError(
                f"Template not found: {template_path}\n"
                f"Expected template to be at: {template_path}"
            )

        with open(template_path, "r", encoding="utf-8") as f:
            self.template_text = f.read()
            self.template = Template(self.template_text)

    def build(self, dataset_sample: DatasetSample) -> LLMPrompt:
        """Build an LLMPrompt from a dataset sample.

        Extracts the judging_sample and passes its attributes to the template:
        - query: Query text from judging_sample
        - document: Document text from judging_sample

        Args:
            dataset_sample: DatasetSample containing judging_sample and context

        Returns:
            LLMPrompt containing the dataset_sample and rendered prompt text

        Raises:
            Exception: If template rendering fails (unrecoverable error)
        """
        # Extract judging_sample from dataset_sample
        judging_sample = dataset_sample.judging_sample

        # Pass JudgingSample model attributes directly to template
        template_vars = {
            "query": judging_sample.query.query_text,
            "document": judging_sample.document.doc_text,
        }

        # Render template
        prompt_text = self.template.render(**template_vars)

        # Create LLMPrompt with dataset_sample and rendered text
        return LLMPrompt.create(
            dataset_sample=dataset_sample,
            prompt_text=prompt_text
        )

    def get_template_text(self) -> str:
        """Get the raw Jinja template text.

        Returns:
            Raw template string (unrendered Jinja template)
        """
        return self.template_text
