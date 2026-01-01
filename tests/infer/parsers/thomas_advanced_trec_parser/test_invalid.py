"""Invalid output tests for ThomasAdvancedTrecParser.

Tests verify proper error handling for malformed, invalid,
or out-of-spec responses that should fail parsing.
"""

from __future__ import annotations
import pytest

from llm_ensemble.infer.adapters.driven.parsers.thomas_advanced_trec_parser import ThomasAdvancedTrecParser
from llm_ensemble.infer.domain.entities.parse_issues import ParserIssueCode


@pytest.fixture
def parser():
    """Create ThomasAdvancedTrecParser instance for testing."""
    return ThomasAdvancedTrecParser()


@pytest.mark.unit
def test_parse_invalid_score_negative(parser: ThomasAdvancedTrecParser):
    """Parse JSON with negative score - should fail validation."""
    raw_text = '{"M": 2, "T": 1, "O": -1}'

    score, issues = parser.parse(raw_text)

    assert score is None
    assert len(issues) == 1
    assert issues[0].code == ParserIssueCode.INVALID_FIELD_VALUE


@pytest.mark.unit
def test_parse_invalid_score_out_of_range(parser: ThomasAdvancedTrecParser):
    """Parse JSON with score 4 - out of valid range (0-3)."""
    raw_text = '{"M": 2, "T": 1, "O": 4}'

    score, issues = parser.parse(raw_text)

    assert score is None
    assert len(issues) == 1
    assert issues[0].code == ParserIssueCode.INVALID_FIELD_VALUE


@pytest.mark.unit
def test_parse_invalid_score_too_high(parser: ThomasAdvancedTrecParser):
    """Parse JSON with score 10 - way out of range."""
    raw_text = '{"M": 2, "T": 1, "O": 10}'

    score, issues = parser.parse(raw_text)

    assert score is None
    assert len(issues) == 1
    assert issues[0].code == ParserIssueCode.INVALID_FIELD_VALUE


@pytest.mark.unit
def test_parse_missing_o_field(parser: ThomasAdvancedTrecParser):
    """Parse JSON missing required O field."""
    raw_text = '{"M": 2, "T": 1}'

    score, issues = parser.parse(raw_text)

    assert score is None
    assert len(issues) == 1
    assert issues[0].code == ParserIssueCode.MISSING_REQUIRED_FIELD


@pytest.mark.unit
def test_parse_null_o_field(parser: ThomasAdvancedTrecParser):
    """Parse JSON with null O field."""
    raw_text = '{"M": 2, "T": 1, "O": null}'

    score, issues = parser.parse(raw_text)

    assert score is None
    assert len(issues) >= 1
    assert any(i.code == ParserIssueCode.INVALID_FIELD_VALUE for i in issues)


@pytest.mark.unit
def test_parse_no_json_found(parser: ThomasAdvancedTrecParser):
    """Parse plain text with no JSON structure or recognizable pattern."""
    raw_text = "This is just some random text without any scores."

    score, issues = parser.parse(raw_text)

    assert score is None
    assert len(issues) == 1
    assert issues[0].code == ParserIssueCode.MALFORMED_RESPONSE


@pytest.mark.unit
def test_parse_float_instead_of_int(parser: ThomasAdvancedTrecParser):
    """Parse JSON with float value instead of integer."""
    raw_text = '{"M": 2, "T": 1, "O": 2.5}'

    score, issues = parser.parse(raw_text)

    assert score is None
    assert len(issues) == 1
    assert issues[0].code == ParserIssueCode.INVALID_FIELD_VALUE


@pytest.mark.unit
def test_parse_empty_string(parser: ThomasAdvancedTrecParser):
    """Parse empty string input."""
    raw_text = ""

    score, issues = parser.parse(raw_text)

    assert score is None
    assert len(issues) == 1
    assert issues[0].code == ParserIssueCode.MALFORMED_RESPONSE


@pytest.mark.unit
def test_parse_string_score_invalid(parser: ThomasAdvancedTrecParser):
    """Parse JSON with non-numeric string score."""
    raw_text = '{"M": 2, "T": 1, "O": "abc"}'

    score, issues = parser.parse(raw_text)

    assert score is None
    assert len(issues) >= 1
    assert any(i.code == ParserIssueCode.INVALID_FIELD_VALUE for i in issues)
