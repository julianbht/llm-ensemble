"""Thomas et al. advanced prompt builder.

Prompt builder that uses the Thomas et al. template with advanced features enabled.
Enables role description and aspects-based evaluation in the prompt.
"""

from __future__ import annotations
from jinja2 import Template

from llm_ensemble.ingest.schemas.dataset_sample import DatasetSample
from llm_ensemble.infer.ports import PromptBuilder
from llm_ensemble.libs.runtime.path_manager import PathManager


class ThomasBuilderAdvanced(PromptBuilder):
    """Advanced Thomas et al. prompt builder with role and aspects enabled.

    Loads the thomas-et-al-prompt.jinja template and enables advanced features:
    - role: True (shows search quality rater role description)
    - aspects: True (enables aspects-based evaluation with M, T, O scores)

    Passes query and document text directly to the template.
    """

    def __init__(self, prompt_name: str, template_path: str, prompt_template_id):
        """Initialize advanced Thomas et al. prompt builder.

        Loads the template from templates/thomas-advanced.jinja using PathManager.
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
        """Build prompt from dataset sample with advanced features (pure building logic).

        Extracts the judging_sample and passes query/document text to the template.
        Uses thomas-advanced.jinja which includes role description and M, T, O scoring.

        Args:
            dataset_sample: DatasetSample containing judging_sample and context

        Returns:
            Tuple of (dataset_sample, prompt_text)

        Raises:
            Exception: If template rendering fails (unrecoverable error)
        """
        # Extract judging_sample from dataset_sample
        judging_sample = dataset_sample.judging_sample

        # Pass query/document text to template
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
