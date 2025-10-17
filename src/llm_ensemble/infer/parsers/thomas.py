"""Response parser for thomas-et-al prompt format.

Handles parsing of JSON-formatted LLM judge outputs.
"""

from __future__ import annotations
import json
import re
from typing import Optional, Callable


# Type alias for parser functions
ParserFunction = Callable[[str], tuple[Optional[int], list[str]]]


def parse_thomas_response(raw_text: str) -> tuple[Optional[int], list[str]]:
    """Parse thomas-et-al prompt response to extract relevance label.

    The template expects JSON output in one of two formats:
    - Simple: {"O": 2}
    - Multi-aspect: {"M": 2, "T": 1, "O": 1}

    Where O is the final relevance score (0, 1, or 2).

    Args:
        raw_text: Raw text response from the LLM

    Returns:
        Tuple of (label, warnings):
        - label: Extracted O score (0, 1, or 2), or None if parsing failed
        - warnings: List of warning messages for parsing issues

    Example:
        >>> parse_thomas_response('{"O": 2}')
        (2, [])
        >>> parse_thomas_response('The answer is {"O": 1} based on...')
        (1, [])
        >>> parse_thomas_response('invalid')
        (None, ['Failed to parse JSON from response'])
    """
    warnings = []

    # Try to find JSON object in the response
    # Look for patterns like {"O": N} or {"M": N, "T": N, "O": N}
    json_match = re.search(r'\{[^}]*"O"\s*:\s*\d+[^}]*\}', raw_text)

    if not json_match:
        warnings.append("No JSON object with 'O' field found in response")
        return None, warnings

    json_str = json_match.group(0)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        warnings.append(f"Failed to parse JSON: {e}")
        return None, warnings

    # Extract the O score
    o_score = data.get("O")

    if o_score is None:
        warnings.append("Missing 'O' field in parsed JSON")
        return None, warnings

    # Validate the score is 0, 1, or 2
    if not isinstance(o_score, int) or o_score not in [0, 1, 2]:
        warnings.append(f"Invalid O score: {o_score} (expected 0, 1, or 2)")
        return None, warnings

    return o_score, warnings


def load_parser(parser_name: str) -> ParserFunction:
    """Load a response parser function by name.

    Args:
        parser_name: Name of the parser function (e.g., "parse_thomas_response")

    Returns:
        The parser function

    Raises:
        ValueError: If the parser function doesn't exist

    Example:
        >>> parser = load_parser("parse_thomas_response")
        >>> label, warnings = parser('{"O": 2}')
        >>> label
        2
    """
    # Get all functions in this module
    import sys
    current_module = sys.modules[__name__]

    # Try to get the parser function
    if not hasattr(current_module, parser_name):
        # List available parsers (functions that start with 'parse_')
        available = [
            name for name in dir(current_module)
            if name.startswith('parse_') and callable(getattr(current_module, name))
        ]
        raise ValueError(
            f"Parser '{parser_name}' not found. "
            f"Available parsers: {', '.join(available)}"
        )

    parser_func = getattr(current_module, parser_name)

    # Verify it's callable
    if not callable(parser_func):
        raise ValueError(f"'{parser_name}' is not a callable parser function")

    return parser_func
