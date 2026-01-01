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
    assert warnings[0].code == ParserIssueCode.VALIDATION_ISSUE


@pytest.mark.unit
def test_parse_invalid_score_negative(parser: ThomasAdvancedParser):
    """Parse JSON with negative score - should fail validation."""
    raw_text = '{"M": 2, "T": 1, "O": -1}'

    score, warnings = parser.parse(raw_text)

    assert score is None
    assert len(warnings) == 1
    assert warnings[0].code == ParserIssueCode.VALIDATION_ISSUE


@pytest.mark.unit
def test_parse_invalid_score_out_of_range(parser: ThomasAdvancedParser):
    """Parse JSON with score 4 - out of valid range."""
    raw_text = '{"M": 2, "T": 1, "O": 4}'

    score, warnings = parser.parse(raw_text)

    assert score is None
    assert len(warnings) == 1
    assert warnings[0].code == ParserIssueCode.VALIDATION_ISSUE


@pytest.mark.unit
def test_parse_no_json_found(parser: ThomasAdvancedParser):
    """Parse plain text with no JSON structure."""
    raw_text = "I think this document is relevant to the query."

    score, warnings = parser.parse(raw_text)

    assert score is None
    assert len(warnings) == 1
    assert warnings[0].code == ParserIssueCode.PARSE_ISSUE


@pytest.mark.unit
def test_parse_float_instead_of_int(parser: ThomasAdvancedParser):
    """Parse JSON with float value instead of integer."""
    raw_text = '{"M": 2, "T": 1, "O": 1.5}'

    score, warnings = parser.parse(raw_text)

    assert score is None
    assert len(warnings) == 1
    assert warnings[0].code == ParserIssueCode.VALIDATION_ISSUE

@pytest.mark.unit
def test_parse_empty_string(parser: ThomasAdvancedParser):
    """Parse empty string input."""
    raw_text = ""

    score, warnings = parser.parse(raw_text)

    assert score is None
    assert len(warnings) == 1
    assert warnings[0].code == ParserIssueCode.PARSE_ISSUE
