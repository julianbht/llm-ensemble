"""Structured warning system for infer CLI ports.

This module defines port-specific warning classes with codes for analytics.
Each port has its own warning class and code enum to track issues during inference.

Warning codes are kept general to avoid taxonomy explosion. Specific details
should be captured in the message and metadata fields, not as separate codes.

Design principles:
- Reader port: Excluded (failures should crash before inference starts)
- Writer port: Excluded (failures should raise exceptions)
- Provider/Parser/Prompt: Full warning coverage for observability
- Codes enable grouping, filtering, and analytics in downstream analysis
- Keep codes GENERAL - use metadata for specifics
"""

from __future__ import annotations
from abc import ABC
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


# ============================================================================
# Base Warning (Abstract)
# ============================================================================


class BaseWarning(BaseModel, ABC):
    """Abstract base class for all warning types.

    All port-specific warnings inherit from this base to enable:
    - Polymorphic collections (list[BaseWarning])
    - Shared helper functions (warning_to_dict, warnings_summary)
    - Consistent structure across all warning types

    Each concrete warning class pairs a code enum with structured metadata.
    """

    code: str | Enum = Field(
        ...,
        description="Structured warning code for grouping/filtering"
    )

    message: str = Field(
        ...,
        description="Human-readable warning message with context"
    )

    metadata: dict[str, str | int | float] = Field(
        default_factory=dict,
        description="Optional metadata for analytics (e.g., retry_count, field_name)"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert warning to dict for serialization."""
        return {
            "type": self.__class__.__name__,
            "code": self.code.value if isinstance(self.code, Enum) else self.code,
            "message": self.message,
            "metadata": self.metadata,
        }


# ============================================================================
# Provider Warnings (LLMProvider port)
# ============================================================================


class ProviderWarningCode(str, Enum):
    """Warning codes for LLM provider issues.

    General categories to avoid taxonomy explosion. Use metadata for specifics.
    """

    # not sure about this one as its not descriptive / analytical
    API_ERROR = "api_error"  # Network, timeout, rate limit, API failures 
    EMPTY_RESPONSE = "empty_response"
    MALFORMED_RESPONSE = "malformed_response"
    RETRY_FAILED = "retry_failed"  # Exhausted retries or fallback used
    INVALID_PARAMETER = "invalid_parameter"  # Invalid parameters, unsupported features
    OTHER = "other"


class ProviderWarning(BaseWarning):
    """Warning from LLM provider adapter.

    Tracks issues during API communication, retries, and response generation.

    Example:
        >>> ProviderWarning(
        ...     code=ProviderWarningCode.API_ERROR,
        ...     message="Rate limited, retried after 2s backoff",
        ...     metadata={"retry_count": 3, "backoff_seconds": 2}
        ... )
    """

    code: ProviderWarningCode  # Override base type with specific enum


# ============================================================================
# Parser Warnings (ResponseParser port)
# ============================================================================


class ParserWarningCode(str, Enum):
    """Warning codes for response parser issues.

    General categories to avoid taxonomy explosion. Use metadata for specifics.
    """

    PARSE_ERROR = "parse_error"  # Malformed JSON/XML, invalid format
    FIELD_ERROR = "field_error"  # Missing, wrong type, invalid value
    VALIDATION_ERROR = "validation_error"  # Out of range, failed constraints
    PARTIAL_PARSE = "partial_parse"  # Some fields extracted, others missing
    OTHER = "other"


class ParserWarning(BaseWarning):
    """Warning from response parser adapter.

    Tracks issues during parsing of raw LLM responses into structured LLMScore.

    Example:
        >>> ParserWarning(
        ...     code=ParserWarningCode.FIELD_ERROR,
        ...     message="Missing 'confidence' field in JSON response",
        ...     metadata={"field_name": "confidence", "expected_type": "float"}
        ... )
    """

    code: ParserWarningCode  # Override base type with specific enum


# ============================================================================
# Prompt Warnings (PromptBuilder port)
# ============================================================================


class PromptWarningCode(str, Enum):
    """Warning codes for prompt builder issues.

    General categories to avoid taxonomy explosion. Use metadata for specifics.
    """

    RENDERING_ERROR = "rendering_error"  # Rendering errors, missing variables
    VALIDATION_ERROR = "validation_error"  # Exceeds limits, failed constraints
    OTHER = "other"


class PromptWarning(BaseWarning):
    """Warning from prompt builder adapter.

    Tracks issues during prompt construction from templates and samples.

    Example:
        >>> PromptWarning(
        ...     code=PromptWarningCode.RENDERING_ERROR,
        ...     message="Variable 'rationale_instructions' not found, using empty string",
        ...     metadata={"variable_name": "rationale_instructions"}
        ... )
    """

    code: PromptWarningCode  # Override base type with specific enum


# ============================================================================
# Utility functions
# ============================================================================


def warning_to_string(warning: BaseWarning) -> str:
    """Convert structured warning to string format.

    Format: "[TYPE:CODE] message"

    Args:
        warning: Structured warning object

    Returns:
        String representation for logging and display

    Example:
        >>> w = ParserWarning(code=ParserWarningCode.FIELD_ERROR, message="No label field")
        >>> warning_to_string(w)
        '[ParserWarning:field_error] No label field'
    """
    code_value = warning.code.value if isinstance(warning.code, Enum) else warning.code
    return f"[{warning.__class__.__name__}:{code_value}] {warning.message}"


def warnings_to_dict_list(warnings: list[BaseWarning]) -> list[dict[str, Any]]:
    """Convert list of warnings to list of dicts for serialization.

    Args:
        warnings: List of warning objects

    Returns:
        List of dictionaries suitable for JSON serialization

    Example:
        >>> warnings = [
        ...     ParserWarning(code=ParserWarningCode.FIELD_ERROR, message="Missing field"),
        ...     ProviderWarning(code=ProviderWarningCode.API_ERROR, message="Timeout")
        ... ]
        >>> warnings_to_dict_list(warnings)
        [{'type': 'ParserWarning', 'code': 'field_error', ...}, ...]
    """
    return [w.to_dict() for w in warnings]


def warnings_summary(warnings: list[BaseWarning]) -> dict[str, int]:
    """Generate summary statistics of warnings by type.

    Args:
        warnings: List of warning objects

    Returns:
        Dictionary mapping warning class name to count

    Example:
        >>> warnings = [
        ...     ParserWarning(code=ParserWarningCode.FIELD_ERROR, message="Missing field"),
        ...     ParserWarning(code=ParserWarningCode.PARSE_ERROR, message="Bad JSON"),
        ...     ProviderWarning(code=ProviderWarningCode.API_ERROR, message="Timeout")
        ... ]
        >>> warnings_summary(warnings)
        {'ParserWarning': 2, 'ProviderWarning': 1}
    """
    summary: dict[str, int] = {}
    for warning in warnings:
        warning_type = warning.__class__.__name__
        summary[warning_type] = summary.get(warning_type, 0) + 1
    return summary
