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
    The parse() method wraps parse_raw() and maps DTO to LLMScore domain objects.
    """

    def __init__(self, parser_spec_id: UUID):
        """Initialize parser with parser spec ID for domain object creation.

        Args:
            parser_spec_id: UUID of the parser spec entity (for LLMScore UUID computation)
        """
        self.parser_spec_id = parser_spec_id

    def parse(self, raw_text: str) -> LLMScore:
        """Parse LLM response to extract structured relevance score.

        Calls parse_raw() to get DTO, then maps to LLMScore domain object.

        Args:
            raw_text: Raw text response from the LLM

        Returns:
            LLMScore with llm_response_text set to raw_text and extracted
            label/confidence/rationale and warnings. Parsed fields may be None
            if parsing failed.

        Note:
            The returned LLMScore must always include llm_response_text (set to raw_text).
            If parsing completely fails, return LLMScore with llm_response_text set
            and all parsed fields as None with appropriate warnings.
        """
        # Call adapter's pure parsing logic
        dto = self.parse_raw(raw_text)
        
        # Map DTO to domain object
        return LLMScore.create(
            llm_response_text=dto.llm_response_text,
            parser_spec_id=self.parser_spec_id,
            label=dto.label,
            confidence=dto.confidence,
            rationale=dto.rationale,
            warnings=dto.warnings,
        )

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
