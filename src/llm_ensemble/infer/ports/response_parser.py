"""Port interface for response parsers.

Defines the abstract contract that all response parser adapters must implement.
This allows the domain/providers to depend on an abstraction rather than concrete
parser implementations.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional


class ResponseParser(ABC):
    """Abstract base class for response parsers.

    All response parser adapters must inherit from this class and implement
    the parse() method.

    Example:
        >>> class ThomasResponseParser(ResponseParser):
        ...     def parse(self, raw_text):
        ...         # Extract JSON and parse
        ...         return (label, warnings)
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
            Exception: If critical parsing error occurs (not expected format)
        """
        pass
