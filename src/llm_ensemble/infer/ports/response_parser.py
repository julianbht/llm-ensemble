"""Port interface for response parsers.

Defines the abstract contract that all response parser adapters must implement.
This allows the system to parse different LLM output formats without coupling
to specific parsing implementations.

Parser identity (parser_name) and configuration come from config.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

from llm_ensemble.infer.schemas.llm_judgement import LLMScore


class ResponseParser(ABC):
    """Abstract base class for response parsers.

    Implementations can parse different output formats (JSON, XML, plain text)
    while providing a consistent interface to extract structured relevance scores.

    The parser extracts an LLMScore from raw text. If parsing fails, it should
    return an LLMScore with None fields and warnings explaining the failure.

    Parser identity (parser_name) and configuration come from config and are
    passed to constructor.
    """

    def __init__(self, parser_name: str, score_field: str = "O"):
        """Initialize response parser with identity from config.

        Args:
            parser_name: Natural key for Parser entity (from config)
            score_field: Field name to extract score from (from config, default: 'O')
        """
        self.parser_name = parser_name
        self.score_field = score_field

    @abstractmethod
    def parse(self, raw_text: str) -> LLMScore:
        """Parse LLM response to extract structured relevance score.

        Args:
            raw_text: Raw text response from the LLM

        Returns:
            LLMScore with llm_response_text set to raw_text and extracted
            label/confidence/rationale and warnings. Parsed fields may be None
            if parsing failed. Warnings are included in LLMScore.warnings for
            any parsing issues encountered.

        Note:
            The returned LLMScore must always include llm_response_text (set to raw_text).
            If parsing completely fails, return LLMScore with llm_response_text set
            and all parsed fields as None with appropriate warnings.
            Never raise exceptions - always return a result.
        """
        pass
