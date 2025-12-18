"""Builder for response parser adapters.

Simple, explicit mapping of parser names to adapter classes.
No decorators, no hidden registration - just a clear dictionary.

To add a new parser:
1. Create adapter class that extends ResponseParser port
2. Import it here
3. Add to PARSERS dict
"""

from __future__ import annotations
from typing import Dict, Type

from llm_ensemble.infer.application.ports.driven.response_parser_port import ResponseParserPort
from llm_ensemble.infer.adapters.parsers.thomas_simple_parser import (
    ThomasSimpleParser,
)


# Explicit mapping of parser names to adapter classes
PARSERS: Dict[str, Type[ResponseParserPort]] = {
    "thomas-simple": ThomasSimpleParser,
}


class ParserAdapterFactory:
    """Builder for creating response parser instances."""

    @staticmethod
    def create(parser_name: str) -> ResponseParserPort:
        """Build and return a parser adapter instance.

        Args:
            parser_name: Name of the parser (e.g., 'thomas-simple')

        Returns:
            Instantiated parser adapter

        Raises:
            ValueError: If parser not found
        """
        if parser_name not in PARSERS:
            available = ", ".join(sorted(PARSERS.keys()))
            raise ValueError(
                f"Parser '{parser_name}' not found. "
                f"Available: {available}"
            )

        adapter_class = PARSERS[parser_name]
        return adapter_class(parser_name=parser_name)

    @staticmethod
    def list_available() -> list[str]:
        """List all available parser names.

        Returns:
            Sorted list of parser names
        """
        return sorted(PARSERS.keys())

    @staticmethod
    def has_parser(parser_name: str) -> bool:
        """Check if parser is available.

        Args:
            parser_name: Name of the parser

        Returns:
            True if parser exists
        """
        return parser_name in PARSERS
