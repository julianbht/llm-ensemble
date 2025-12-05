"""Port interface for response parsers.

Defines the abstract contract that all response parser adapters must implement.
Adapters translate raw LLM response text into domain LLMScore entities.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

from llm_ensemble.infer.schemas.entities.llm_score import LLMScore


class ResponseParserPort(ABC):
    """Abstract interface for response parsers.

    Adapters implement this interface to build LLMScore domain entities
    from raw LLM response text. The adapter is responsible for:
    1. Parsing the response text (internal implementation detail)
    2. Constructing and returning complete LLMScore entities

    This follows proper hexagonal architecture - adapters (outer layer)
    depend on domain entities (inner layer), translating external concerns
    (parsing logic, format handling) into domain concepts the service can work with.
    """

    @abstractmethod
    def parse(self, raw_text: str) -> LLMScore:
        """Parse LLM response and create LLMScore domain entity.

        Extracts structured data from the raw response text and constructs
        an LLMScore domain entity with all necessary context (parser metadata,
        extracted fields, warnings).

        Args:
            raw_text: Raw text response from the LLM

        Returns:
            LLMScore domain entity with parsed fields and parser metadata.
            All parsed fields may be None if parsing failed.
            Never raises exceptions - always returns a result with warnings.
        """
        pass
