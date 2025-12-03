"""Jinja2-based prompt builder adapter.

Prompt builder that uses the Thomas et al. template for relevance judging.
The template is colocated with this adapter in the templates/ subdirectory.
"""

from __future__ import annotations
from jinja2 import Template

from llm_ensemble.ingest.schemas.dataset_sample import DatasetSample
from llm_ensemble.infer.ports import PromptBuilderPort
from llm_ensemble.libs.runtime.path_manager import PathManager


class JinjaPromptBuilder(PromptBuilderPort):
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

    def __init__(self, prompt_name: str, template_path: str, prompt_template_id):
        """Initialize Jinja prompt builder.

        Loads the template from templates/thomas-simple.jinja using PathManager.
        """
        super().__init__(prompt_name, template_path, prompt_template_id)
        
        # Load template using PathManager
        full_template_path = PathManager.get_prompt_templates_dir() / template_path

        if not full_template_path.exists():
            raise FileNotFoundError(
                f"Template not found: {full_template_path}\n"
                f"Expected template to be at: {full_template_path}"
            )

        with open(full_template_path, "r", encoding="utf-8") as f:
            self.template_text = f.read()
            self.template = Template(self.template_text)

    def build_raw(self, dataset_sample: DatasetSample) -> tuple[DatasetSample, str]:
        """Build prompt from dataset sample (pure building logic).

        Extracts the judging_sample and passes its attributes to the template:
        - query: Query text from judging_sample
        - document: Document text from judging_sample

        Args:
            dataset_sample: DatasetSample containing judging_sample and context

        Returns:
            Tuple of (dataset_sample, prompt_text)

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

        return dataset_sample, prompt_text

    def get_template_text(self) -> str:
        """Get the raw Jinja template text.

        Returns:
            Raw template string (unrendered Jinja template)
        """
        return self.template_text
