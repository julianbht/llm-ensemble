"""Port interface for prompt builders.

Defines the abstract contract that all prompt builder adapters must implement.
This allows the system to work with different prompt formats and templates
without coupling to specific implementations.

Adapters implement build_raw() returning (dataset_sample, prompt_text) tuple (pure logic).
The port provides build() that maps tuple to LLMPrompt domain objects.

Prompt identity (prompt_name) and template path come from config.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from uuid import UUID

from llm_ensemble.ingest.schemas.dataset_sample import DatasetSample
from llm_ensemble.infer.schemas.llm_judgement import LLMPrompt


class PromptBuilder(ABC):
    """Abstract base class for prompt builders.

    Implementations can build prompts using different templates and formats
    while providing a consistent interface to the LLM provider adapters.

    Adapters implement build_raw() returning (dataset_sample, prompt_text) tuple.
    Adapters are pure implementations - they don't know their own identity.
    """

    @abstractmethod
    def build_raw(self, dataset_sample: DatasetSample) -> tuple[DatasetSample, str]:
        """Build prompt from dataset sample (pure adapter logic).

        Adapters implement this method with pure building logic.
        Returns tuple without creating domain objects.

        Args:
            dataset_sample: DatasetSample containing judging_sample and context

        Returns:
            Tuple of (dataset_sample, prompt_text)

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
