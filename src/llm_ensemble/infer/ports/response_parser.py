"""Port interface for response parsers.

Defines the abstract contract that all response parser adapters must implement.
Each parser is tightly coupled to its specific prompt and knows what to look for.

Adapters implement parse_raw() returning ParsedScoreDTO (pure logic).
The port provides parse() that maps DTO to LLMScore domain objects.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from uuid import UUID

from llm_ensemble.infer.schemas.llm_judgement import LLMScore
from llm_ensemble.infer.schemas.parsed_score_dto import ParsedScoreDTO


class ResponseParser(ABC):
    """Abstract base class for response parsers.

    Parsers are tightly coupled to specific prompts and know exactly
    what format to expect and what fields to extract.

    Adapters implement parse_raw() returning ParsedScoreDTO (pure parsing logic).
    Adapter identity (parser_name) comes from builder and is stored in adapter.
    """

    def __init__(self, parser_name: str):
        """Initialize response parser with identity.

        Args:
            parser_name: Natural key for parser identity (from builder)
        """
        self.parser_name = parser_name

    @abstractmethod
    def parse_raw(self, raw_text: str) -> ParsedScoreDTO:
        """Parse LLM response to extract structured data (pure adapter logic).

        Adapters implement this method with pure parsing logic.
        Returns DTO without creating domain objects.

        Args:
            raw_text: Raw text response from the LLM

        Returns:
            ParsedScoreDTO with extracted fields and warnings.
            All parsed fields may be None if parsing failed.
            Never raises exceptions - always returns a result with warnings.
        """
        pass
