"""Port interface for prompt builders.

Defines the abstract contract that all prompt builder adapters must implement.
Adapters translate template rendering concerns into domain LLMPrompt entities.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

from llm_ensemble.ingest.schemas.dataset_sample import DatasetSample
from llm_ensemble.infer.schemas.entities.llm_prompt import LLMPrompt


class PromptBuilderPort(ABC):
    """Abstract interface for prompt builders.

    Adapters implement this interface to build LLMPrompt domain entities
    from DatasetSample objects. The adapter is responsible for:
    1. Rendering prompt text from template (internal implementation detail)
    2. Constructing and returning complete LLMPrompt entities

    This follows proper hexagonal architecture - adapters (outer layer)
    depend on domain entities (inner layer), translating external concerns
    (templates, rendering) into domain concepts the service can work with.
    """

    @abstractmethod
    def build_prompt(self, dataset_sample: DatasetSample) -> LLMPrompt:
        """Build complete LLMPrompt entity from dataset sample.

        Renders the prompt text and constructs an LLMPrompt domain entity
        with all necessary context (template metadata, sample, rendered text).

        Args:
            dataset_sample: DatasetSample containing judging_sample and context

        Returns:
            LLMPrompt domain entity ready for inference

        Raises:
            KeyError: If template variables missing from dataset
            Exception: For unrecoverable rendering errors
        """
        pass
