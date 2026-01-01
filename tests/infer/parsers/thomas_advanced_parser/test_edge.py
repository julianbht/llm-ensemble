"""Edge case tests for ThomasAdvancedParser.

Tests verify parser behavior with unusual but potentially
valid inputs, boundary conditions, and metadata access.
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
def test_parse_multiple_json_objects(parser: ThomasAdvancedParser):
    """Parse text with multiple JSON objects - should extract first match."""
    raw_text = '''
    First attempt: {"M": 0, "T": 0, "O": 0}
    Second attempt: {"M": 2, "T": 2, "O": 2}
    '''

    score, warnings = parser.parse(raw_text)

    # Should extract first valid JSON
    assert score is not None
    assert score.label == RelevanceScore.IRRELEVANT
    assert len(warnings) == 0


@pytest.mark.unit
def test_parse_json_with_extra_fields(parser: ThomasAdvancedParser):
    """Parse JSON with additional fields beyond M, T, O."""
    raw_text = '{"M": 2, "T": 1, "O": 1, "explanation": "Good match"}'

    score, warnings = parser.parse(raw_text)

    assert score is not None
    assert score.label == RelevanceScore.RELEVANT
    assert len(warnings) == 0


@pytest.mark.unit
def test_get_parser_metadata(parser: ThomasAdvancedParser):
    """Verify parser metadata is correctly configured."""
    metadata = parser.get_parser()

    assert metadata.name == "thomas-advanced"
    assert metadata.version == "1.0"
    assert metadata.id is not None
