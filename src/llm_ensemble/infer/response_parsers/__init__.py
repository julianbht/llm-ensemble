"""Response parsers for LLM judge outputs.

Each parser module should export a `parse()` function that takes:
- raw_text: str (raw LLM response)

And returns a tuple of (label, warnings):
- label: Optional[int] (extracted relevance score or None if parsing failed)
- warnings: list[str] (parsing warnings/errors)
"""

from __future__ import annotations
import importlib
from pathlib import Path


def load_parser(parser_name: str):
    """Load a response parser module by name.

    Args:
        parser_name: Name of the parser module (e.g., "thomas")

    Returns:
        The parser module with a `parse()` function

    Raises:
        ImportError: If the parser module doesn't exist
        AttributeError: If the module doesn't have a `parse()` function

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

    if not hasattr(module, "parse"):
        raise AttributeError(
            f"Parser module '{parser_name}' must define a parse() function"
        )

    return module


__all__ = ["load_parser"]
