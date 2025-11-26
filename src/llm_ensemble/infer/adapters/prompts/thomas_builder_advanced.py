"""Thomas et al. advanced prompt builder.

Prompt builder that uses the Thomas et al. template with advanced features enabled.
Enables role description and aspects-based evaluation in the prompt.
"""

from __future__ import annotations
from jinja2 import Template

from llm_ensemble.ingest.schemas.dataset_sample import DatasetSample
from llm_ensemble.infer.ports import PromptBuilder
from llm_ensemble.infer.schemas.llm_judgement import LLMPrompt
from llm_ensemble.libs.runtime.path_manager import PathManager


class ThomasBuilderAdvanced(PromptBuilder):
    """Advanced Thomas et al. prompt builder with role and aspects enabled.

    Loads the thomas-et-al-prompt.jinja template and enables advanced features:
    - role: True (shows search quality rater role description)
    - aspects: True (enables aspects-based evaluation with M, T, O scores)

    Passes query and document text directly to the template.
    """

    def __init__(self):
        """Initialize advanced Thomas et al. prompt builder.

        Loads the template from templates/thomas-advanced.jinja using PathManager.
        """
        # Load template using PathManager
        template_path = PathManager.get_prompt_templates_dir() / "thomas-advanced.jinja"

        if not template_path.exists():
            raise FileNotFoundError(
                f"Template not found: {template_path}\n"
                f"Expected template to be at: {template_path}"
            )

        with open(template_path, "r", encoding="utf-8") as f:
            self.template_text = f.read()
            self.template = Template(self.template_text)

    def build(self, dataset_sample: DatasetSample) -> LLMPrompt:
        """Build an LLMPrompt from a dataset sample with advanced features.

        Extracts the judging_sample and passes query/document text to the template.
        Uses thomas-advanced.jinja which includes role description and M, T, O scoring.

        Args:
            dataset_sample: DatasetSample containing judging_sample and context

        Returns:
            LLMPrompt containing the dataset_sample and rendered prompt text

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
