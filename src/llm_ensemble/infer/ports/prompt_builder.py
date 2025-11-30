"""Port interface for prompt builders.

Defines the abstract contract that all prompt builder adapters must implement.
This allows the system to work with different prompt formats and templates
without coupling to specific implementations.

Prompt identity (prompt_name) and template path come from config.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

from llm_ensemble.ingest.schemas.dataset_sample import DatasetSample
from llm_ensemble.infer.schemas.llm_judgement import LLMPrompt


class PromptBuilder(ABC):
    """Abstract base class for prompt builders.

    Implementations can build prompts using different templates and formats
    while providing a consistent interface to the LLM provider adapters.

    Returns an LLMPrompt containing the dataset_sample and rendered prompt_text.

    Prompt identity (prompt_name) and template path come from config and are
    passed to constructor.
    """

    def __init__(self, prompt_name: str, template_path: str):
        """Initialize prompt builder with identity from config.

        Args:
            prompt_name: Natural key for Prompt entity (from config)
            template_path: Path to template file relative to templates dir (from config)
        """
        self.prompt_name = prompt_name
        self.template_path = template_path

    @abstractmethod
    def build(self, dataset_sample: DatasetSample) -> LLMPrompt:
        """Build an LLMPrompt from a dataset sample.

        Args:
            dataset_sample: DatasetSample containing judging_sample and context

        Returns:
            LLMPrompt containing the dataset_sample and rendered prompt text

        Raises:
            Exception: For unrecoverable errors (e.g., template file missing).
        """
        pass

    @abstractmethod
    def get_template_text(self) -> str:
        """Get the raw template text for database storage.

        Returns the unrendered template string (e.g., Jinja template source).
        This enables storing templates in the database for analysis and filtering.

        Returns:
            Raw template text (before variable substitution)
        """
        pass
