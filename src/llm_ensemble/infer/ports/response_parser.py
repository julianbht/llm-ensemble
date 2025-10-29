"""Port interface for response parsers.

Defines the abstract contract that all response parser adapters must implement.
This allows the system to parse different LLM output formats without coupling
to specific parsing implementations.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

from llm_ensemble.infer.schemas.llm_score import LLMScore


class ResponseParser(ABC):
    """Abstract base class for response parsers.

    Implementations can parse different output formats (JSON, XML, plain text)
    while providing a consistent interface to extract structured relevance scores.

    The parser extracts an LLMScore from raw text. If parsing fails, it should
    return an LLMScore with None fields and warnings explaining the failure.

    Example:
        >>> class JsonResponseParser(ResponseParser):
        ...     def parse(self, raw_text):
        ...         try:
        ...             data = json.loads(raw_text)
        ...             return LLMScore(
        ...                 label=data.get("label"),
        ...                 confidence=data.get("confidence"),
        ...                 rationale=data.get("rationale")
        ...             )
        ...         except Exception as e:
        ...             return LLMScore()  # All fields None = parse failure
    """

    @abstractmethod
    def parse(self, raw_text: str) -> tuple[LLMScore, list[str]]:
        """Parse LLM response to extract structured relevance score.

        Args:
            raw_text: Raw text response from the LLM

        Returns:
            Tuple of (score, warnings):
            - score: LLMScore with extracted label/confidence/rationale (fields may be None if parsing failed)
            - warnings: List of warning messages for parsing issues

        Note:
            If parsing completely fails, return LLMScore() with all None fields
            and appropriate warnings. Never raise exceptions - always return a result.
        """
        pass
