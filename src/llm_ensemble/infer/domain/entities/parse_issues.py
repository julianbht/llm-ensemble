"""Structured parser issue tracking for infer CLI.

Tracks data quality issues during LLM response parsing.
Codes enable grouping, filtering, and analytics in downstream analysis.

Design principles:
- Keep codes GENERAL - use metadata for specifics to avoid taxonomy explosion
- Codes for analytics and filtering, messages for human readability
- Metadata for additional context (field names, attempted values, etc.)
"""

from __future__ import annotations
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class ParserIssueCode(str, Enum):
    """Issue codes for response parser problems.

    Codes are verbose and specific to make analytics and debugging clear.
    Use metadata for additional context (field names, attempted values, etc.).
    """

    MALFORMED_RESPONSE = "malformed_response"
    """Response format is malformed or unparseable (e.g., invalid JSON, no recognizable structure)."""

    MISSING_REQUIRED_FIELD = "missing_required_field"
    """Required field is missing from response (e.g., missing 'O' score field)."""

    INVALID_FIELD_VALUE = "invalid_field_value"
    """Field value is invalid (e.g., out of range, wrong type, null when required)."""

    NON_STANDARD_FORMAT = "non_standard_format"
    """Response uses non-standard format requiring extraction (e.g., markdown, embedded JSON, regex patterns)."""

    TYPE_COERCION_APPLIED = "type_coercion_applied"
    """Value type was coerced to expected type (e.g., string "2" converted to int 2)."""

    LOW_CONFIDENCE_EXTRACTION = "low_confidence_extraction"
    """Score extracted using fuzzy/heuristic matching (e.g., keyword matching, low reliability)."""

    OTHER = "other"
    """Uncategorized issue."""


class ParserIssue(BaseModel):
    """Issue encountered during LLM response parsing.

    Tracks problems when parsing raw LLM responses into structured LLMScore.
    Used for diagnostics and data quality analysis.
    """

    code: ParserIssueCode = Field(
        ...,
        description="Structured issue code for grouping/filtering"
    )

    message: str = Field(
        ...,
        description="Human-readable issue description with context"
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata for analytics (e.g., field_name, attempted_value)"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert issue to dict for serialization."""
        return {
            "code": self.code.value,
            "message": self.message,
            "metadata": self.metadata,
        }


# ============================================================================
# Utility functions
# ============================================================================


def issue_to_string(issue: ParserIssue) -> str:
    """Convert structured parser issue to string format.

    Format: "[CODE] message"

    Args:
        issue: Structured parser issue object

    Returns:
        String representation for logging and display
    """
    return f"[{issue.code.value}] {issue.message}"


def issues_to_dict_list(issues: list[ParserIssue]) -> list[dict[str, Any]]:
    """Convert list of parser issues to list of dicts for serialization.

    Args:
        issues: List of parser issue objects

    Returns:
        List of dictionaries suitable for JSON serialization
    """
    return [i.to_dict() for i in issues]


def issues_summary(issues: list[ParserIssue]) -> dict[str, int]:
    """Generate summary statistics of parser issues by code.

    Args:
        issues: List of parser issue objects

    Returns:
        Dictionary mapping issue code to count
    """
    summary: dict[str, int] = {}
    for issue in issues:
        code = issue.code.value
        summary[code] = summary.get(code, 0) + 1
    return summary
