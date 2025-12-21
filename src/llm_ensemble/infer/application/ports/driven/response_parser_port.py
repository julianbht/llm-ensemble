"""Port interface for response parsers.

Defines the abstract contract that all response parser adapters must implement.
Adapters translate raw LLM response text into domain LLMScore entities.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

from llm_ensemble.infer.domain.entities.llm_score import LLMScore
from llm_ensemble.infer.domain.entities.reponse_parser import ResponseParser
from llm_ensemble.infer.schemas.warnings import BaseWarning


class ResponseParserPort(ABC):
    """Abstract interface for response parsers.

    Adapters implement this interface to build LLMScore domain entities
    from raw LLM response text. The adapter is responsible for:
    1. Parsing the response text (internal implementation detail)
    2. Constructing and returning complete LLMScore entities
    3. Providing metadata about the parser (via get_parser())

    This follows proper hexagonal architecture - adapters (outer layer)
    depend on domain entities (inner layer), translating external concerns
    (parsing logic, format handling) into domain concepts the service can work with.
    """

    @abstractmethod
    def parse(self, raw_text: str) -> tuple[LLMScore, list[BaseWarning]]:
        """Parse LLM response and create LLMScore domain entity.

        Extracts structured data from the raw response text and constructs
        an LLMScore domain entity with parsed fields.

        Args:
            raw_text: Raw text response from the LLM

        Returns:
            Tuple of (LLMScore, warnings):
            - LLMScore: domain entity with parsed fields (no parser metadata)
            - warnings: List of parser warnings from the parsing process
            All parsed fields may be None if parsing failed.
            Never raises exceptions - always returns a result with warnings.
        """
        pass

    @abstractmethod
    def get_parser(self) -> ResponseParser:
        """Get Parser metadata for this adapter.

        Returns:
            Parser entity with id and name
        """
        pass
