"""Port interface for response parsers.

Defines the abstract contract that all response parser adapters must implement.
This allows the system to parse different LLM output formats without coupling
to specific parsing implementations.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional


class ResponseParser(ABC):
    """Abstract base class for response parsers.

    Implementations can parse different output formats (JSON, XML, plain text)
    while providing a consistent interface to the LLM provider adapters.

    Example:
        >>> class JsonResponseParser(ResponseParser):
        ...     def parse(self, raw_text):
        ...         data = json.loads(raw_text)
        ...         return data.get("O"), []
    """

    @abstractmethod
    def parse(self, raw_text: str) -> tuple[Optional[int], list[str]]:
        """Parse LLM response to extract relevance label.

        Args:
            raw_text: Raw text response from the LLM

        Returns:
            Tuple of (label, warnings):
            - label: Extracted relevance score (0, 1, or 2), or None if parsing failed
            - warnings: List of warning messages for parsing issues

        Raises:
            ValueError: If raw_text is invalid or malformed
        """
        pass
