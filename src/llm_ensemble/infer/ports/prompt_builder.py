"""Port interface for prompt builders.

Defines the abstract contract that all prompt builder adapters must implement.
This allows the system to work with different prompt formats and templates
without coupling to specific implementations.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

from llm_ensemble.ingest.schemas import JudgingSample
from llm_ensemble.infer.schemas.llm_request import LLMRequest


class PromptBuilder(ABC):
    """Abstract base class for prompt builders.

    Implementations can build prompts using different templates and formats
    while providing a consistent interface to the LLM provider adapters.

    Returns LLMRequest containing the rendered prompt and any warnings from
    template rendering (missing variables, validation errors, etc.).

    Example:
        >>> class JinjaPromptBuilder(PromptBuilder):
        ...     def build(self, example):
        ...         prompt = self.template.render(
        ...             query=example.query_text,
        ...             page_text=example.doc
        ...         )
        ...         return LLMRequest(prompt=prompt, warnings=[])
    """

    @abstractmethod
    def build(self, example: JudgingSample) -> LLMRequest:
        """Build a prompt from a judging example.

        Args:
            example: JudgingSample object containing query and document

        Returns:
            LLMRequest containing rendered prompt and any warnings from building

        Raises:
            Exception: Only for unrecoverable errors (e.g., template file missing).
                       Recoverable issues should be captured as warnings in the returned LLMRequest.
        """
        pass
