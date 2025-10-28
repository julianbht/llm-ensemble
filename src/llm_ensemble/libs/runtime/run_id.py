"""Run ID generation utilities.

Provides functions for generating unique, timestamped run identifiers.
"""

from __future__ import annotations
from datetime import datetime


def generate_run_id(name_hint: str) -> str:
    """Generate a unique timestamped run ID.

    Format: YYYYMMDD_HHMMSS_{sanitized_hint}

    The name_hint is sanitized to remove special characters and limited to 30 characters.

    Args:
        name_hint: Hint for the run (e.g., dataset name, model name)

    Returns:
        Unique run ID string

    Example:
        >>> generate_run_id("gpt-oss-20b")
        '20250128_143022_gpt-oss-20b'
        >>> generate_run_id("my dataset with spaces & special!")
        '20250128_143022_mydatasetwithspacesspecial'
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sanitize name_hint (remove special chars, limit length)
    safe_hint = "".join(c for c in name_hint if c.isalnum() or c in "-_")[:30]
    return f"{timestamp}_{safe_hint}"
