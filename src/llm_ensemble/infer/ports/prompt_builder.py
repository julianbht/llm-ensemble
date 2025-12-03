"""Port interface for prompt builders.

Defines the abstract contract that all prompt builder adapters must implement.
This allows the system to work with different prompt formats and templates
without coupling to specific implementations.

Uses template method pattern: build() is concrete and handles tuple→Domain mapping,
subclasses implement build_raw() with pure building logic.

Adapter identity (prompt_name) comes from builder and is used to create entities.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

from llm_ensemble.ingest.schemas.dataset_sample import DatasetSample
from llm_ensemble.infer.schemas.llm_judgement import LLMPrompt


class PromptBuilder(ABC):
    """Abstract base class for prompt builders with built-in domain mapping.

    Implementations provide building logic in build_raw(), which returns a simple tuple.
    The base class handles conversion to LLMPrompt domain objects.

    Template Method Pattern:
    - build() (concrete): calls build_raw() and creates domain object
    - build_raw() (abstract): subclasses implement building logic

    This separates pure template logic from domain object creation.
    Prompt identity (prompt_name) comes from builder and is passed to constructor.
    """

    def __init__(self, prompt_name: str):
        """Initialize prompt builder with identity.

        Args:
            prompt_name: Natural key for prompt identity (from builder)
        """
        self.prompt_name = prompt_name

    @abstractmethod
    def build_raw(self, dataset_sample: DatasetSample) -> tuple[DatasetSample, str]:
        """Build prompt from dataset sample (pure adapter logic).

        Subclasses implement this with pure building logic.

        Args:
            dataset_sample: DatasetSample containing judging_sample and context

        Returns:
            Tuple of (dataset_sample, prompt_text)

        Raises:
            Exception: For unrecoverable errors (e.g., template file missing).
        """
        pass

    def build(self, dataset_sample: DatasetSample) -> LLMPrompt:
        """Build prompt and create LLMPrompt domain object.

        Public interface called by service. Internally calls build_raw()
        and maps result to domain object.

        Args:
            dataset_sample: DatasetSample containing judging_sample and context

        Returns:
            LLMPrompt domain object with prompt text and identity
        """
        # Call subclass implementation (returns tuple)
        ds, prompt_text = self.build_raw(dataset_sample)

        # Compute UUID from prompt name
        from llm_ensemble.libs.db import compute_prompt_template_uuid
        prompt_template_id = compute_prompt_template_uuid(self.prompt_name)

        # Map to domain entity (port layer's responsibility)
        return LLMPrompt.create(
            dataset_sample=ds,
            prompt_text=prompt_text,
            prompt_template_id=prompt_template_id,
        )

    @abstractmethod
    def get_template_text(self) -> str:
        """Get the raw template text for database storage.

        Returns the unrendered template string (e.g., Jinja template source).
        This enables storing templates in the database for analysis and filtering.

        Returns:
            Raw template text (before variable substitution)
        """
        pass
