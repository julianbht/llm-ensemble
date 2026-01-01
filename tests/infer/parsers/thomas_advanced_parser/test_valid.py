"""Valid output tests for ThomasAdvancedParser.

Tests verify successful parsing of well-formed JSON responses
matching the thomas-advanced prompt format: {"M": N, "T": N, "O": N}
"""

from __future__ import annotations
import pytest

from llm_ensemble.infer.adapters.driven.parsers.thomas_advanced_parser import ThomasAdvancedParser
from llm_ensemble.libs.schemas.relevance_score import RelevanceScore


@pytest.fixture
def parser():
    """Create ThomasAdvancedParser instance for testing."""
    return ThomasAdvancedParser()


@pytest.mark.unit
def test_parse_perfect_json_score_0(parser: ThomasAdvancedParser):
    """Parse perfect JSON with score 0 (irrelevant)."""
    raw_text = '{"M": 0, "T": 0, "O": 0}'

    score, warnings = parser.parse(raw_text)

    assert score is not None
    assert score.label == RelevanceScore.IRRELEVANT
    assert score.confidence is None
    assert score.rationale is None
    assert len(warnings) == 0


@pytest.mark.unit
def test_parse_perfect_json_score_1(parser: ThomasAdvancedParser):
    """Parse perfect JSON with score 1 (relevant)."""
    raw_text = '{"M": 1, "T": 1, "O": 1}'

    score, warnings = parser.parse(raw_text)

    assert score is not None
    assert score.label == RelevanceScore.RELEVANT
    assert len(warnings) == 0


@pytest.mark.unit
def test_parse_perfect_json_score_2(parser: ThomasAdvancedParser):
    """Parse perfect JSON with score 2 (highly relevant)."""
    raw_text = '{"M": 2, "T": 2, "O": 2}'

    score, warnings = parser.parse(raw_text)

    assert score is not None
    assert score.label == RelevanceScore.HIGHLY_RELEVANT
    assert len(warnings) == 0


@pytest.mark.unit
def test_parse_json_with_extra_whitespace(parser: ThomasAdvancedParser):
    """Parse JSON with extra whitespace around fields and values."""
    raw_text = '{ "M" : 2 , "T" : 1 , "O" : 1 }'

    score, warnings = parser.parse(raw_text)

    assert score is not None
    assert score.label == RelevanceScore.RELEVANT
    assert len(warnings) == 0


@pytest.mark.unit
def test_parse_json_embedded_in_text(parser: ThomasAdvancedParser):
    """Parse JSON embedded within explanatory text."""
    raw_text = '''
    Based on my analysis, the scores are {"M": 2, "T": 1, "O": 1}.
    This reflects good intent match but moderate trust.
    '''

    score, warnings = parser.parse(raw_text)

    assert score is not None
    assert score.label == RelevanceScore.RELEVANT
    assert len(warnings) == 0


@pytest.mark.unit
def test_parse_json_different_field_order(parser: ThomasAdvancedParser):
    """Parse JSON with fields in different order than M, T, O."""
    raw_text = '{"T": 1, "O": 2, "M": 2}'

    score, warnings = parser.parse(raw_text)

    assert score is not None
    assert score.label == RelevanceScore.HIGHLY_RELEVANT
    assert len(warnings) == 0


@pytest.mark.unit
def test_parse_json_with_different_mt_values(parser: ThomasAdvancedParser):
    """Parse JSON where M and T differ from O (realistic scenario)."""
    raw_text = '{"M": 2, "T": 0, "O": 1}'

    score, warnings = parser.parse(raw_text)

    assert score is not None
    assert score.label == RelevanceScore.RELEVANT
    assert len(warnings) == 0
