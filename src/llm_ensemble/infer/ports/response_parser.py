"""Port interface for response parsers.

Defines the abstract contract that all response parser adapters must implement.
Each parser is tightly coupled to its specific prompt and knows what to look for.

Uses template method pattern: parse() is concrete and handles DTO→Domain mapping,
subclasses implement parse_raw() with pure parsing logic.

Adapter identity (parser_name and parser_spec_id) comes from builder.
"""

from __future__ import annotations
import uuid
from abc import ABC, abstractmethod
from uuid import UUID

from llm_ensemble.infer.schemas.entities.llm_score import LLMScore
from llm_ensemble.infer.schemas.parsed_score_dto import ParsedScoreDTO


class ResponseParserPort(ABC):
    """Abstract base class for response parsers with built-in domain mapping.

    Implementations provide parsing logic in parse_raw(), which returns a simple DTO.
    The base class handles conversion to LLMScore domain objects.

    Template Method Pattern:
    - parse() (concrete): calls parse_raw() and creates domain object
    - parse_raw() (abstract): subclasses implement parsing logic

    This separates pure parsing logic from domain object creation.
    Parser identity (parser_name and parser_spec_id) comes from builder.
    """

    def __init__(self, parser_name: str, parser_spec_id: UUID | None = None):
        """Initialize response parser with identity.

        Args:
            parser_name: Natural key for parser identity (from builder)
            parser_spec_id: UUID for this parser spec (random if None)
        """
        self.parser_name = parser_name
        self.parser_spec_id = parser_spec_id or uuid.uuid4()

    @abstractmethod
    def parse_raw(self, raw_text: str) -> ParsedScoreDTO:
        """Parse LLM response to extract structured data (pure adapter logic).

        Subclasses implement this method with pure parsing logic.

        Args:
            raw_text: Raw text response from the LLM

        Returns:
            ParsedScoreDTO with extracted fields and warnings.
            All parsed fields may be None if parsing failed.
            Never raises exceptions - always returns a result with warnings.
        """
        pass

    def parse(self, raw_text: str) -> LLMScore:
        """Parse LLM response and create LLMScore domain object.

        Public interface called by service. Internally calls parse_raw()
        and maps result to domain object.

        Args:
            raw_text: Raw text response from the LLM

        Returns:
            LLMScore domain object with parsed fields, parser, and identity
        """
        # Call subclass implementation (returns DTO)
        parsed_dto = self.parse_raw(raw_text)

        # Import here to avoid circular dependency
        from llm_ensemble.infer.schemas.entities.parser import Parser

        # Create parser object
        parser = Parser(
            id=self.parser_spec_id,
            name=self.parser_name,
        )

        # Map to domain entity (port layer's responsibility)
        return LLMScore(
            parser=parser,
            llm_response_text=parsed_dto.llm_response_text,
            label=parsed_dto.label,
            confidence=parsed_dto.confidence,
            rationale=parsed_dto.rationale,
            warnings=parsed_dto.warnings,
        )
