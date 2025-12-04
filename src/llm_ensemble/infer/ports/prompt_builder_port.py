"""Port interface for prompt builders.

Defines the abstract contract that all prompt builder adapters must implement.
This allows the system to work with different prompt formats and templates
without coupling to specific implementations.

Uses template method pattern: build() is concrete and handles tuple→Domain mapping,
subclasses implement build_raw() with pure building logic.

Adapter identity (prompt_name and prompt_template_id) comes from builder.
"""

from __future__ import annotations
import uuid
from abc import ABC, abstractmethod
from uuid import UUID

from llm_ensemble.ingest.schemas.dataset_sample import DatasetSample
from llm_ensemble.infer.schemas.entities.llm_prompt import LLMPrompt


class PromptBuilderPort(ABC):
    """Abstract base class for prompt builders with built-in domain mapping.

    Implementations provide building logic in build_raw(), which returns a simple tuple.
    The base class handles conversion to LLMPrompt domain objects.

    Template Method Pattern:
    - build() (concrete): calls build_raw() and creates domain object
    - build_raw() (abstract): subclasses implement building logic

    This separates pure template logic from domain object creation.
    Prompt identity (prompt_name and prompt_template_id) comes from builder.
    """

    def __init__(self, prompt_name: str, prompt_template_id: UUID | None = None):
        """Initialize prompt builder with identity.

        Args:
            prompt_name: Natural key for prompt identity (from builder)
            prompt_template_id: UUID for this prompt template (random if None)
        """
        self.prompt_name = prompt_name
        self.prompt_template_id = prompt_template_id or uuid.uuid4()

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
            LLMPrompt domain object with prompt text, template, and identity
        """
        # Call subclass implementation (returns tuple)
        ds, prompt_text = self.build_raw(dataset_sample)

        # Import here to avoid circular dependency
        from llm_ensemble.infer.schemas.entities.prompt_template import PromptTemplate

        # Create prompt template object
        prompt_template = PromptTemplate(
            id=self.prompt_template_id,
            name=self.prompt_name,
            template_text=self.get_template_text(),
        )

        # Map to domain entity (port layer's responsibility)
        return LLMPrompt(
            prompt_template=prompt_template,
            dataset_sample=ds,
            prompt_text=prompt_text,
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
