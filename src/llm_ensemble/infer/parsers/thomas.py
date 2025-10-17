"""Response parser for thomas-et-al prompt format.

Handles parsing of JSON-formatted LLM judge outputs.
"""

from __future__ import annotations
import json
import re
from typing import Optional


def parse(raw_text: str) -> tuple[Optional[int], list[str]]:
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
        >>> parse('{"O": 2}')
        (2, [])
        >>> parse('The answer is {"O": 1} based on...')
        (1, [])
        >>> parse('invalid')
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


