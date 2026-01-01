"""Invalid output tests for ThomasAdvancedParser.

Tests verify proper error handling for malformed, invalid,
or out-of-spec responses that should fail parsing.
"""

from __future__ import annotations
import pytest

from llm_ensemble.infer.adapters.driven.parsers.thomas_advanced_parser import ThomasAdvancedParser
from llm_ensemble.infer.domain.entities.parse_issues import ParserIssueCode


@pytest.fixture
def parser():
    """Create ThomasAdvancedParser instance for testing."""
    return ThomasAdvancedParser()


@pytest.mark.unit
def test_parse_invalid_score_3(parser: ThomasAdvancedParser):
    """Parse JSON with score 3 - out of range for thomas-advanced (expects 0-2 only)."""
    raw_text = '{"M": 2, "T": 1, "O": 3}'

    score, warnings = parser.parse(raw_text)

    assert score is None
    assert len(warnings) == 1
    assert warnings[0].code == ParserIssueCode.VALIDATION_ERROR
    assert "expected 0, 1, or 2" in warnings[0].message


@pytest.mark.unit
def test_parse_invalid_score_negative(parser: ThomasAdvancedParser):
    """Parse JSON with negative score - should fail validation."""
    raw_text = '{"M": 2, "T": 1, "O": -1}'

    score, warnings = parser.parse(raw_text)

    assert score is None
    assert len(warnings) == 1
    assert warnings[0].code == ParserIssueCode.VALIDATION_ERROR
    assert "Invalid O score" in warnings[0].message
    assert warnings[0].metadata["field_name"] == "O"


@pytest.mark.unit
def test_parse_invalid_score_out_of_range(parser: ThomasAdvancedParser):
    """Parse JSON with score 4 - out of valid range."""
    raw_text = '{"M": 2, "T": 1, "O": 4}'

    score, warnings = parser.parse(raw_text)

    assert score is None
    assert len(warnings) == 1
    assert warnings[0].code == ParserIssueCode.VALIDATION_ERROR
    assert "expected 0, 1, or 2" in warnings[0].message


@pytest.mark.unit
def test_parse_missing_o_field(parser: ThomasAdvancedParser):
    """Parse JSON missing the required O field."""
    raw_text = '{"M": 2, "T": 1}'

    score, warnings = parser.parse(raw_text)

    assert score is None
    assert len(warnings) >= 1
    # Should have FIELD_ERROR for missing O
    field_errors = [w for w in warnings if w.code == ParserIssueCode.FIELD_ERROR]
    assert len(field_errors) > 0
    assert "Missing 'O' field" in field_errors[0].message


@pytest.mark.unit
def test_parse_invalid_json_syntax(parser: ThomasAdvancedParser):
    """Parse malformed JSON with missing closing brace."""
    raw_text = '{"M": 2, "T": 1, "O": 1'

    score, warnings = parser.parse(raw_text)

    assert score is None
    assert len(warnings) >= 1
    parse_errors = [w for w in warnings if w.code == ParserIssueCode.PARSE_ERROR]
    assert len(parse_errors) > 0


@pytest.mark.unit
def test_parse_no_json_found(parser: ThomasAdvancedParser):
    """Parse plain text with no JSON structure."""
    raw_text = "I think this document is relevant to the query."

    score, warnings = parser.parse(raw_text)

    assert score is None
    assert len(warnings) == 1
    assert warnings[0].code == ParserIssueCode.PARSE_ERROR
    assert "No JSON object with 'M', 'T', 'O' fields found" in warnings[0].message


@pytest.mark.unit
def test_parse_string_instead_of_int(parser: ThomasAdvancedParser):
    """Parse JSON with string value instead of integer."""
    raw_text = '{"M": 2, "T": 1, "O": "1"}'

    score, warnings = parser.parse(raw_text)

    assert score is None
    assert len(warnings) == 1
    assert warnings[0].code == ParserIssueCode.VALIDATION_ERROR
    assert "Invalid O score" in warnings[0].message


@pytest.mark.unit
def test_parse_float_instead_of_int(parser: ThomasAdvancedParser):
    """Parse JSON with float value instead of integer."""
    raw_text = '{"M": 2, "T": 1, "O": 1.5}'

    score, warnings = parser.parse(raw_text)

    assert score is None
    assert len(warnings) == 1
    assert warnings[0].code == ParserIssueCode.VALIDATION_ERROR


@pytest.mark.unit
def test_parse_null_value(parser: ThomasAdvancedParser):
    """Parse JSON with null value for O field."""
    raw_text = '{"M": 2, "T": 1, "O": null}'

    score, warnings = parser.parse(raw_text)

    assert score is None
    assert len(warnings) >= 1
    field_errors = [w for w in warnings if w.code == ParserIssueCode.FIELD_ERROR]
    assert len(field_errors) > 0
