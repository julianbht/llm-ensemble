"""Run ID generation utilities.

Provides functions for generating unique, timestamped run identifiers.
"""

from __future__ import annotations
from datetime import datetime
from typing import Optional


def generate_run_id(name_hints: Optional[list[str]] = None) -> str:
    """Generate a unique timestamped run ID, optionally with name hints.

    Format:
    - With hints: YYYYMMDD_HHMMSS_{hint1}_{hint2}_{hint3}
    - Without hints: YYYYMMDD_HHMMSS

    Each name_hint is sanitized to remove special characters and limited to 30 characters.
    Hints are joined with underscores.

    Args:
        name_hints: Optional list of hints from configs (e.g., ['gpt20b', 'thomas', 'ndjson']).
                   Only hints explicitly provided by configs should be included (no fallbacks).

    Returns:
        Unique run ID string

    Examples:
        >>> generate_run_id(['gpt20b', 'thomas', 'ndjson'])
        '20250128_143022_gpt20b_thomas_ndjson'
        >>> generate_run_id(['llmjudge'])
        '20250128_143022_llmjudge'
        >>> generate_run_id([])
        '20250128_143022'
        >>> generate_run_id(None)
        '20250128_143022'
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # If no hints provided or empty list, return just timestamp
    if not name_hints:
        return timestamp

    # Sanitize each hint (remove special chars, limit length)
    safe_hints = [
        "".join(c for c in hint if c.isalnum() or c in "-_")[:30]
        for hint in name_hints
        if hint  # Filter out empty strings
    ]

    # Filter out any hints that became empty after sanitization
    safe_hints = [h for h in safe_hints if h]

    # If all hints were filtered out, return just timestamp
    if not safe_hints:
        return timestamp

    # Join hints with underscores
    hints_str = "_".join(safe_hints)
    return f"{timestamp}_{hints_str}"
