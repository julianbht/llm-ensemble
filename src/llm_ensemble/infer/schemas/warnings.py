"""Structured warning system for infer CLI ports.

This module defines port-specific warning classes with codes for analytics.
Each port has its own warning class and code enum to track issues during inference.

Warning codes are kept general to avoid taxonomy explosion. Specific details
should be captured in the message and metadata fields, not as separate codes.

Design principles:
- Reader port: Excluded (failures should crash before inference starts)
- Writer port: Minimal warnings (rare failure cases only)
- Provider/Parser/Prompt: Full warning coverage for observability
- Codes enable grouping, filtering, and analytics in downstream analysis
- Keep codes GENERAL - use metadata for specifics
"""

from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field


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


class ProviderWarning(BaseModel):
    """Warning from LLM provider adapter.

    Tracks issues during API communication, retries, and response generation.

    Example:
        >>> ProviderWarning(
        ...     code=ProviderWarningCode.API_ERROR,
        ...     message="Rate limited, retried after 2s backoff",
        ...     metadata={"retry_count": 3, "backoff_seconds": 2}
        ... )
    """

    code: ProviderWarningCode = Field(
        ...,
        description="Structured warning code for grouping/filtering"
    )

    message: str = Field(
        ...,
        description="Human-readable warning message with context"
    )

    metadata: dict[str, str | int | float] = Field(
        default_factory=dict,
        description="Optional metadata for analytics (e.g., retry_count, error_type)"
    )


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


class ParserWarning(BaseModel):
    """Warning from response parser adapter.

    Tracks issues during parsing of raw LLM responses into structured LLMScore.

    Example:
        >>> ParserWarning(
        ...     code=ParserWarningCode.FIELD_ERROR,
        ...     message="Missing 'confidence' field in JSON response",
        ...     metadata={"field_name": "confidence", "expected_type": "float"}
        ... )
    """

    code: ParserWarningCode = Field(
        ...,
        description="Structured warning code for grouping/filtering"
    )

    message: str = Field(
        ...,
        description="Human-readable warning message with context"
    )

    metadata: dict[str, str | int | float] = Field(
        default_factory=dict,
        description="Optional metadata for analytics (e.g., field_name, expected_type)"
    )


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


class PromptWarning(BaseModel):
    """Warning from prompt builder adapter.

    Tracks issues during prompt construction from templates and samples.

    Example:
        >>> PromptWarning(
        ...     code=PromptWarningCode.RENDERING_ERROR,
        ...     message="Variable 'rationale_instructions' not found, using empty string",
        ...     metadata={"variable_name": "rationale_instructions"}
        ... )
    """

    code: PromptWarningCode = Field(
        ...,
        description="Structured warning code for grouping/filtering"
    )

    message: str = Field(
        ...,
        description="Human-readable warning message with context"
    )

    metadata: dict[str, str | int | float] = Field(
        default_factory=dict,
        description="Optional metadata for analytics (e.g., variable_name, content_length)"
    )


# ============================================================================
# Writer Warnings (JudgementWriter port) - MINIMAL
# ============================================================================


class WriterWarningCode(str, Enum):
    """Warning codes for judgement writer issues.

    Kept minimal - most writer issues should raise exceptions.
    """

    SERIALIZATION_ERROR = "serialization_error"  # Issues converting to output format
    OTHER = "other"


class WriterWarning(BaseModel):
    """Warning from judgement writer adapter.

    Kept minimal - writer issues usually crash. Only for recoverable edge cases.

    Example:
        >>> WriterWarning(
        ...     code=WriterWarningCode.SERIALIZATION_ERROR,
        ...     message="NaN value replaced with null during JSON serialization",
        ...     metadata={"field": "confidence", "original_value": "NaN"}
        ... )
    """

    code: WriterWarningCode = Field(
        ...,
        description="Structured warning code for grouping/filtering"
    )

    message: str = Field(
        ...,
        description="Human-readable warning message with context"
    )

    metadata: dict[str, str | int | float] = Field(
        default_factory=dict,
        description="Optional metadata for analytics"
    )


# ============================================================================
# Utility functions
# ============================================================================


def warning_to_string(warning: ProviderWarning | ParserWarning | PromptWarning | WriterWarning) -> str:
    """Convert structured warning to legacy string format.

    This is a temporary bridge for existing code that expects string warnings.
    Format: "[CODE] message"

    Args:
        warning: Structured warning object

    Returns:
        String representation for backward compatibility

    Example:
        >>> w = ParserWarning(code=ParserWarningCode.FIELD_ERROR, message="No label field")
        >>> warning_to_string(w)
        '[field_error] No label field'
    """
    return f"[{warning.code.value}] {warning.message}"
