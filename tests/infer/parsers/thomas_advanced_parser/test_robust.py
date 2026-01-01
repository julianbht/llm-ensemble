"""Robust parsing tests for ThomasAdvancedParser.

Tests verify the parser can handle messy LLM outputs and still extract scores.
These cases should succeed with PARTIAL_PARSE warnings rather than fail completely.

Note: O field is critical - if missing or null, parsing must fail.
M and T fields are optional for extraction purposes (we only care about O).
"""

from __future__ import annotations
import pytest

from llm_ensemble.infer.adapters.driven.parsers.thomas_advanced_parser import ThomasAdvancedParser
from llm_ensemble.infer.domain.entities.parse_issues import ParserIssueCode
from llm_ensemble.libs.schemas.relevance_score import RelevanceScore


@pytest.fixture
def parser():
    """Create ThomasAdvancedParser instance for testing."""
    return ThomasAdvancedParser()


@pytest.mark.unit
def test_parse_string_score_converts_to_int(parser: ThomasAdvancedParser):
    """Parse JSON with string "1" instead of int 1 - should convert and succeed."""
    raw_text = '{"M": 2, "T": 1, "O": "1"}'

    score, warnings = parser.parse(raw_text)

    assert score is not None
    assert score.label == RelevanceScore.RELEVANT
    assert len(warnings) == 1
    assert warnings[0].code == ParserIssueCode.PARTIAL_PARSE


@pytest.mark.unit
def test_parse_malformed_json_missing_brace(parser: ThomasAdvancedParser):
    """Parse malformed JSON with missing closing brace - should fix and extract."""
    raw_text = '{"M": 2, "T": 1, "O": 1'

    score, warnings = parser.parse(raw_text)

    assert score is not None
    assert score.label == RelevanceScore.RELEVANT
    assert len(warnings) >= 1
    assert any(w.code == ParserIssueCode.PARTIAL_PARSE for w in warnings)


@pytest.mark.unit
def test_parse_missing_m_field_only_cares_about_o(parser: ThomasAdvancedParser):
    """Parse JSON missing M field - should still work, we only need O."""
    raw_text = '{"T": 1, "O": 2}'

    score, warnings = parser.parse(raw_text)

    assert score is not None
    assert score.label == RelevanceScore.HIGHLY_RELEVANT
    assert len(warnings) == 0


@pytest.mark.unit
def test_parse_missing_t_field_only_cares_about_o(parser: ThomasAdvancedParser):
    """Parse JSON missing T field - should still work, we only need O."""
    raw_text = '{"M": 2, "O": 1}'

    score, warnings = parser.parse(raw_text)

    assert score is not None
    assert score.label == RelevanceScore.RELEVANT
    assert len(warnings) == 0


@pytest.mark.unit
def test_parse_only_o_field_present(parser: ThomasAdvancedParser):
    """Parse JSON with only O field - should work, that's all we need."""
    raw_text = '{"O": 0}'

    score, warnings = parser.parse(raw_text)

    assert score is not None
    assert score.label == RelevanceScore.IRRELEVANT
    assert len(warnings) == 0


@pytest.mark.unit
def test_parse_extra_text_around_json(parser: ThomasAdvancedParser):
    """Parse JSON embedded in explanation text - should extract score."""
    raw_text = '''
    Let me analyze this query-document pair.
    Here are my scores: {"M": 2, "T": 1, "O": 1}
    In summary, the document is relevant.
    '''

    score, warnings = parser.parse(raw_text)

    assert score is not None
    assert score.label == RelevanceScore.RELEVANT
    assert len(warnings) == 0
