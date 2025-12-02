"""Jinja2-based prompt builder adapter with registry support.

Generic Jinja2 prompt builder that can work with any template.
Template path provided during construction from registry.
"""

from __future__ import annotations
from pathlib import Path
from jinja2 import Template

from llm_ensemble.ingest.schemas.dataset_sample import DatasetSample
from llm_ensemble.infer.ports import PromptBuilder
from llm_ensemble.infer.schemas.llm_judgement import LLMPrompt
from llm_ensemble.infer.adapters.prompts.registry import prompt_registry


@prompt_registry.register(
    name="thomas-simple",
    description="Thomas et al. simple binary relevance prompt",
    template_path="thomas-simple.jinja"
)
class ThomasSimplePromptBuilder(PromptBuilder):
    """Thomas et al. simple prompt (binary relevance scoring).

    Passes JudgingSample model attributes to the template:
    - {{ query }} - The query text
    - {{ document }} - The document text
    """

    def __init__(self, template_path: str):
        """Initialize with template path.

        Args:
            template_path: Path to template file relative to templates dir
        """
        # Load template
        templates_dir = Path(__file__).parent / "templates"
        full_template_path = templates_dir / template_path

        if not full_template_path.exists():
            raise FileNotFoundError(
                f"Template not found: {full_template_path}"
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
