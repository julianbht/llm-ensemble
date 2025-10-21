"""Response parsers for LLM judge outputs.

Each parser module should export a concrete class that implements the
ResponseParser port interface.
"""

from __future__ import annotations
import importlib
from pathlib import Path

from llm_ensemble.infer.ports import ResponseParser


def load_parser(parser_name: str) -> ResponseParser:
    """Load a response parser adapter by name.

    Args:
        parser_name: Name of the parser module (e.g., "thomas")

    Returns:
        An instance of the ResponseParser implementation

    Raises:
        ImportError: If the parser module doesn't exist
        AttributeError: If the module doesn't have the expected class

    Example:
        >>> parser = load_parser("thomas")
        >>> label, warnings = parser.parse('{"O": 2}')
        >>> label
        2
    """
    try:
        module = importlib.import_module(f"llm_ensemble.infer.response_parsers.{parser_name}")
    except ImportError as e:
        # List available parsers
        parsers_dir = Path(__file__).parent
        available = [
            p.stem for p in parsers_dir.glob("*.py")
            if p.stem != "__init__"
        ]
        raise ImportError(
            f"Parser module '{parser_name}' not found. "
            f"Available parsers: {', '.join(available)}"
        ) from e

    # Convert parser_name to class name (e.g., "thomas" -> "ThomasResponseParser")
    class_name = "".join(word.capitalize() for word in parser_name.split("_")) + "ResponseParser"
    
    if not hasattr(module, class_name):
        raise AttributeError(
            f"Parser module '{parser_name}' must define a {class_name} class "
            f"that implements the ResponseParser interface"
        )

    parser_class = getattr(module, class_name)
    return parser_class()


__all__ = ["load_parser"]
