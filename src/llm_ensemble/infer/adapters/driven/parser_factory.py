"""Builder for response parser adapters.

Explicit instantiation of parser adapters with parser-specific constructors.
Each parser adapter defines its own constructor signature and configuration needs.

To add a new parser:
1. Create adapter class that extends ResponseParser port
2. Import it here
3. Add explicit instantiation case in create() method
"""

from __future__ import annotations

from llm_ensemble.infer.application.ports.driven.response_parser_port import ResponseParserPort
from llm_ensemble.infer.adapters.driven.parsers.thomas_simple_parser import ThomasSimpleParser
from llm_ensemble.infer.adapters.driven.parsers.thomas_advanced_parser import ThomasAdvancedParser


class ParserAdapterFactory:
    """Builder for creating response parser instances."""

    @staticmethod
    def create(parser_name: str) -> ResponseParserPort:
        """Build and return a parser adapter instance.

        Uses explicit instantiation per parser to allow parser-specific
        constructor signatures and configuration.

        Args:
            parser_name: Name of the parser (e.g., 'thomas-simple', 'thomas-advanced')

        Returns:
            Instantiated parser adapter

        Raises:
            ValueError: If parser not found
        """
        if parser_name == "thomas-simple":
            return ThomasSimpleParser()
        elif parser_name == "thomas-advanced":
            return ThomasAdvancedParser()
        else:
            available = ", ".join(sorted(["thomas-simple", "thomas-advanced"]))
            raise ValueError(
                f"Parser '{parser_name}' not found. "
                f"Available: {available}"
            )

    @staticmethod
    def list_available() -> list[str]:
        """List all available parser names.

        Returns:
            Sorted list of parser names
        """
        return sorted(["thomas-simple", "thomas-advanced"])

    @staticmethod
    def has_parser(parser_name: str) -> bool:
        """Check if parser is available.

        Args:
            parser_name: Name of the parser

        Returns:
            True if parser exists
        """
        return parser_name in ["thomas-simple", "thomas-advanced"]
