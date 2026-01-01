"""Robust parsing tests for ThomasAdvancedTrecParser.

Tests verify the cascading parser strategy can handle messy LLM outputs:
- Strategy 1: Clean JSON (tests in test_valid.py)
- Strategy 2: Embedded JSON (markdown, text)
- Strategy 3: Regex extraction
- Strategy 4: Fuzzy keyword matching
"""

from __future__ import annotations
import pytest

from llm_ensemble.infer.adapters.driven.parsers.thomas_advanced_trec_parser import ThomasAdvancedTrecParser
from llm_ensemble.infer.domain.entities.parse_issues import ParserIssueCode
from llm_ensemble.libs.schemas.relevance_score import RelevanceScore


@pytest.fixture
def parser():
    """Create ThomasAdvancedTrecParser instance for testing."""
    return ThomasAdvancedTrecParser()


@pytest.mark.unit
def test_parse_markdown_json_block(parser: ThomasAdvancedTrecParser):
    """Parse JSON in markdown code block."""
    raw_text = '''
    Here is my analysis:
    ```json
    {"M": 2, "T": 1, "O": 2}
    ```
    '''

    score, issues = parser.parse(raw_text)

    assert score is not None
    assert score.label == RelevanceScore.HIGHLY_RELEVANT
    assert len(issues) == 1
    assert issues[0].code == ParserIssueCode.PARTIAL_PARSE_ISSUE
    assert "markdown" in issues[0].metadata["extraction_method"]


@pytest.mark.unit
def test_parse_generic_code_block(parser: ThomasAdvancedTrecParser):
    """Parse JSON in generic code block without json marker."""
    raw_text = '''
    My scores:
    ```
    {"M": 3, "T": 2, "O": 3}
    ```
    '''

    score, issues = parser.parse(raw_text)

    assert score is not None
    assert score.label == RelevanceScore.PERFECTLY_RELEVANT
    assert len(issues) == 1
    assert issues[0].code == ParserIssueCode.PARTIAL_PARSE_ISSUE


@pytest.mark.unit
def test_parse_json_embedded_in_text(parser: ThomasAdvancedTrecParser):
    """Parse JSON object embedded in explanation text."""
    raw_text = '''
    Let me analyze this query-document pair.
    The document matches the intent well.
    My scores are: {"M": 2, "T": 2, "O": 2}
    In summary, highly relevant.
    '''

    score, issues = parser.parse(raw_text)

    assert score is not None
    assert score.label == RelevanceScore.HIGHLY_RELEVANT
    assert len(issues) == 1
    assert "embedded_json" in issues[0].metadata["extraction_method"]


@pytest.mark.unit
def test_parse_multiple_json_objects(parser: ThomasAdvancedTrecParser):
    """Parse text with multiple JSON objects - extracts first valid one."""
    raw_text = '''
    First attempt: {"M": 0, "T": 0, "O": 0}
    Let me reconsider: {"M": 3, "T": 3, "O": 3}
    '''

    score, issues = parser.parse(raw_text)

    assert score is not None
    # Should extract first valid JSON
    assert score.label == RelevanceScore.IRRELEVANT
    assert len(issues) == 1


# ============================================================================
# Strategy 3: Regex extraction
# ============================================================================

@pytest.mark.unit
def test_parse_regex_colon_format(parser: ThomasAdvancedTrecParser):
    """Parse score using regex: O: 2 format."""
    raw_text = 'The overall score is O: 2'

    score, issues = parser.parse(raw_text)

    assert score is not None
    assert score.label == RelevanceScore.HIGHLY_RELEVANT
    assert len(issues) == 1
    assert issues[0].code == ParserIssueCode.PARTIAL_PARSE_ISSUE
    assert "regex" in issues[0].metadata["extraction_method"]


@pytest.mark.unit
def test_parse_regex_equals_format(parser: ThomasAdvancedTrecParser):
    """Parse score using regex: O = 1 format."""
    raw_text = 'Setting O = 1 for this pair'

    score, issues = parser.parse(raw_text)

    assert score is not None
    assert score.label == RelevanceScore.RELEVANT
    assert len(issues) == 1
    assert "regex" in issues[0].metadata["extraction_method"]


@pytest.mark.unit
def test_parse_regex_no_quotes(parser: ThomasAdvancedTrecParser):
    """Parse score with unquoted field name."""
    raw_text = 'My final score: M: 3, T: 3, O: 3'

    score, issues = parser.parse(raw_text)

    assert score is not None
    assert score.label == RelevanceScore.PERFECTLY_RELEVANT
    assert len(issues) == 1


# ============================================================================
# Strategy 4: Fuzzy keyword matching
# ============================================================================

@pytest.mark.unit
def test_parse_fuzzy_perfectly_relevant(parser: ThomasAdvancedTrecParser):
    """Parse using fuzzy matching for 'perfectly relevant' keywords."""
    raw_text = 'This document is perfectly relevant to the query and contains the exact answer.'

    score, issues = parser.parse(raw_text)

    assert score is not None
    assert score.label == RelevanceScore.PERFECTLY_RELEVANT
    assert len(issues) == 1
    assert issues[0].code == ParserIssueCode.PARTIAL_PARSE_ISSUE
    assert "fuzzy" in issues[0].metadata["extraction_method"]


@pytest.mark.unit
def test_parse_fuzzy_highly_relevant(parser: ThomasAdvancedTrecParser):
    """Parse using fuzzy matching for 'highly relevant' keywords."""
    raw_text = 'The page is highly relevant and contains vital information.'

    score, issues = parser.parse(raw_text)

    assert score is not None
    assert score.label == RelevanceScore.HIGHLY_RELEVANT
    assert len(issues) == 1
    assert "fuzzy" in issues[0].metadata["extraction_method"]


@pytest.mark.unit
def test_parse_fuzzy_relevant(parser: ThomasAdvancedTrecParser):
    """Parse using fuzzy matching for 'relevant' keywords."""
    raw_text = 'This document is relevant and partly helpful for the query.'

    score, issues = parser.parse(raw_text)

    assert score is not None
    assert score.label == RelevanceScore.RELEVANT
    assert len(issues) == 1


@pytest.mark.unit
def test_parse_fuzzy_irrelevant(parser: ThomasAdvancedTrecParser):
    """Parse using fuzzy matching for 'not relevant' keywords."""
    raw_text = 'The document is not relevant to the query at all.'

    score, issues = parser.parse(raw_text)

    assert score is not None
    assert score.label == RelevanceScore.IRRELEVANT
    assert len(issues) == 1


# ============================================================================
# Type conversion and edge cases
# ============================================================================

@pytest.mark.unit
def test_parse_string_score_converts_to_int(parser: ThomasAdvancedTrecParser):
    """Parse JSON with string "2" instead of int 2 - should convert."""
    raw_text = '{"M": 2, "T": 1, "O": "2"}'

    score, issues = parser.parse(raw_text)

    assert score is not None
    assert score.label == RelevanceScore.HIGHLY_RELEVANT
    assert len(issues) == 1
    assert issues[0].code == ParserIssueCode.PARTIAL_PARSE_ISSUE
