"""Jinja2-based prompt builder adapter.

Generic Jinja2 prompt builder that can work with any template.
Template path and prompt name come from config.
"""

from __future__ import annotations
from jinja2 import Template

from llm_ensemble.ingest.schemas.dataset_sample import DatasetSample
from llm_ensemble.infer.ports import PromptBuilder
from llm_ensemble.infer.schemas.llm_judgement import LLMPrompt
from llm_ensemble.libs.runtime.path_manager import PathManager


class JinjaPromptBuilder(PromptBuilder):
    """Generic Jinja2 prompt builder.

    Loads any Jinja2 template from the templates/ directory.
    Template path and prompt name come from config.

    Passes JudgingSample Pydantic model attributes directly to the template:
    - {{ query }} - The query text
    - {{ document }} - The document text

    Template path is relative to the prompt templates directory.
    """

    def __init__(self, prompt_name: str, template_path: str):
        """Initialize Jinja prompt builder.

        Args:
            prompt_name: Natural key for Prompt entity (from config)
            template_path: Path to template file relative to templates dir (from config)
        """
        super().__init__(prompt_name, template_path)

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
