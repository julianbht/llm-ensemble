"""Port interface for prompt builders.

Defines the abstract contract that all prompt builder adapters must implement.
Adapters render prompt strings and provide template metadata.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

from llm_ensemble.ingest.schemas.dataset_sample import DatasetSample
from llm_ensemble.infer.schemas.entities.prompt_template import PromptTemplate


class PromptBuilderPort(ABC):
    """Abstract interface for prompt builders.

    Adapters implement this interface to:
    1. Render prompt text from DatasetSample (pure string transformation)
    2. Provide template metadata as a PromptTemplate entity

    The service layer orchestrates domain entity creation using the
    rendered string and template metadata from the adapter.

    In hexagonal architecture, adapters can depend on domain entities
    (PromptTemplate) - this is the correct dependency direction.
    """

    @property
    @abstractmethod
    def template(self) -> PromptTemplate:
        """Template metadata for this builder.

        Returns:
            PromptTemplate entity with id, name, and template_text
        """
        pass

    @abstractmethod
    def render(self, dataset_sample: DatasetSample) -> str:
        """Render prompt text from dataset sample.

        Extracts variables from dataset_sample and substitutes them
        into the template to produce final prompt text.

        Args:
            dataset_sample: DatasetSample containing judging_sample and context

        Returns:
            Rendered prompt text string

        Raises:
            KeyError: If template variables missing from dataset
            Exception: For unrecoverable rendering errors
        """
        pass
