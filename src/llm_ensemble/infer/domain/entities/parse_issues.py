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

    General categories to avoid taxonomy explosion. Use metadata for specifics.
    """

    PARSE_ISSUE = "parse_issue"  # Malformed JSON/XML, invalid format
    FIELD_ISSUE = "field_issue"  # Missing, wrong type, invalid value
    VALIDATION_ISSUE = "validation_issue"  # Out of range, failed constraints
    PARTIAL_PARSE_ISSUE = "partial_parse_issue"  # Some fields extracted, others missing
    OTHER = "other"


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
