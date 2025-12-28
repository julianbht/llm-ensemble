"""Port interface for prompt builders.

Defines the abstract contract that all prompt builder adapters must implement.
Adapters translate template rendering concerns into prompt text strings.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

from llm_ensemble.ingest.schemas.dataset_sample import DatasetSample
from llm_ensemble.infer.domain.entities.prompt_builder import PromptBuilder


class ForBuildingPrompts(ABC):
    """Abstract interface for prompt builders.

    Adapters implement this interface to render prompt text from templates.
    The adapter is responsible for:
    1. Rendering prompt text from template (internal implementation detail)
    2. Providing metadata about the prompt builder (via get_builder())

    This follows proper hexagonal architecture - adapters (outer layer)
    depend on domain entities (inner layer), translating external concerns
    (templates, rendering) into domain concepts the service can work with.
    """

    @abstractmethod
    def build_prompt(self, dataset_sample: DatasetSample) -> str:
        """Render prompt text from dataset sample.

        Renders the prompt text using the internal template.

        Args:
            dataset_sample: DatasetSample containing judging_sample and context

        Returns:
            Rendered prompt text ready for inference

        Raises:
            KeyError: If template variables missing from dataset
            Exception: For unrecoverable rendering errors
        """
        pass

    @abstractmethod
    def get_builder(self) -> PromptBuilder:
        """Get PromptBuilder metadata for this adapter.

        Returns:
            PromptBuilder entity with id, name, and version
        """
        pass

    @abstractmethod
    def get_template_text(self) -> str:
        """Get the raw template text for this builder.

        Returns:
            Raw template string (unrendered)
        """
        pass
