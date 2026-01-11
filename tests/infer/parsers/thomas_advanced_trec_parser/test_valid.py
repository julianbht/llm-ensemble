"""Valid input tests for ThomasAdvancedTrecParser.

Tests verify parser correctly handles well-formed responses
across all 4 levels of the TREC relevance scale (0-3).
"""

from __future__ import annotations
import pytest

from llm_ensemble.infer.adapters.driven.parsers.thomas_advanced_trec_parser import ThomasAdvancedTrecParser
from llm_ensemble.libs.schemas.relevance_score import RelevanceScore


@pytest.fixture
def parser():
    """Create ThomasAdvancedTrecParser instance for testing."""
    return ThomasAdvancedTrecParser()


@pytest.mark.unit
def test_parse_clean_json_score_0(parser: ThomasAdvancedTrecParser):
    """Parse clean JSON with score 0 (irrelevant)."""
    raw_text = '{"M": 0, "T": 0, "O": 0}'

    score, issue = parser.parse(raw_text)

    assert score is not None
    assert score.label == RelevanceScore.IRRELEVANT
    assert issue is None


@pytest.mark.unit
def test_parse_clean_json_score_1(parser: ThomasAdvancedTrecParser):
    """Parse clean JSON with score 1 (relevant)."""
    raw_text = '{"M": 1, "T": 1, "O": 1}'

    score, issue = parser.parse(raw_text)

    assert score is not None
    assert score.label == RelevanceScore.RELEVANT
    assert issue is None


@pytest.mark.unit
def test_parse_clean_json_score_2(parser: ThomasAdvancedTrecParser):
    """Parse clean JSON with score 2 (highly relevant)."""
    raw_text = '{"M": 2, "T": 2, "O": 2}'

    score, issue = parser.parse(raw_text)

    assert score is not None
    assert score.label == RelevanceScore.HIGHLY_RELEVANT
    assert issue is None


@pytest.mark.unit
def test_parse_clean_json_score_3(parser: ThomasAdvancedTrecParser):
    """Parse clean JSON with score 3 (perfectly relevant)."""
    raw_text = '{"M": 3, "T": 3, "O": 3}'

    score, issue = parser.parse(raw_text)

    assert score is not None
    assert score.label == RelevanceScore.PERFECTLY_RELEVANT
    assert issue is None


@pytest.mark.unit
def test_parse_json_with_whitespace(parser: ThomasAdvancedTrecParser):
    """Parse JSON with extra whitespace."""
    raw_text = '  \n  {"M": 2, "T": 1, "O": 2}  \n  '

    score, issue = parser.parse(raw_text)

    assert score is not None
    assert score.label == RelevanceScore.HIGHLY_RELEVANT
    assert issue is None


@pytest.mark.unit
def test_parse_json_with_extra_fields(parser: ThomasAdvancedTrecParser):
    """Parse JSON with additional fields beyond M, T, O."""
    raw_text = '{"M": 2, "T": 1, "O": 1, "explanation": "Partly relevant"}'

    score, issue = parser.parse(raw_text)

    assert score is not None
    assert score.label == RelevanceScore.RELEVANT
    assert issue is None


@pytest.mark.unit
def test_parse_json_only_o_field(parser: ThomasAdvancedTrecParser):
    """Parse JSON with only O field present."""
    raw_text = '{"O": 3}'

    score, issue = parser.parse(raw_text)

    assert score is not None
    assert score.label == RelevanceScore.PERFECTLY_RELEVANT
    assert issue is None


@pytest.mark.unit
def test_get_parser_metadata(parser: ThomasAdvancedTrecParser):
    """Verify parser metadata is correctly configured."""
    metadata = parser.get_parser()

    assert metadata.name == "thomas-advanced-trec"
    assert metadata.version == "1.0"
    assert metadata.id is not None
