"""Tests for response parser functionality."""

import pytest

from llm_ensemble.infer.response_parsers import load_parser
from llm_ensemble.infer.response_parsers import thomas


@pytest.mark.unit
class TestParseThomas:
    """Tests for thomas parser parse() function."""

    def test_simple_format(self):
        """Test parsing simple {"O": N} format."""
        label, warnings = thomas.parse('{"O": 2}')
        assert label == 2
        assert warnings == []

    def test_multi_aspect_format(self):
        """Test parsing multi-aspect format with M, T, O."""
        label, warnings = thomas.parse('{"M": 1, "T": 2, "O": 1}')
        assert label == 1
        assert warnings == []

    def test_json_in_text(self):
        """Test extracting JSON from surrounding text."""
        label, warnings = thomas.parse(
            'The answer is {"O": 0} based on the analysis.'
        )
        assert label == 0
        assert warnings == []

    def test_invalid_score(self):
        """Test handling invalid O score."""
        label, warnings = thomas.parse('{"O": 5}')
        assert label is None
        assert "Invalid O score" in warnings[0]

    def test_missing_o_field(self):
        """Test handling missing O field."""
        label, warnings = thomas.parse('{"M": 1, "T": 2}')
        assert label is None
        assert "No JSON object with 'O' field found" in warnings[0]

    def test_invalid_json(self):
        """Test handling invalid JSON."""
        label, warnings = thomas.parse('This is not JSON at all')
        assert label is None
        assert len(warnings) > 0


@pytest.mark.unit
class TestLoadParser:
    """Tests for load_parser function."""

    def test_load_thomas_parser(self):
        """Test loading thomas parser module."""
        parser = load_parser("thomas")
        assert hasattr(parser, "parse")
        assert callable(parser.parse)

        # Verify it works
        label, warnings = parser.parse('{"O": 2}')
        assert label == 2
        assert warnings == []

    def test_load_nonexistent_parser(self):
        """Test error handling for nonexistent parser."""
        with pytest.raises(ImportError) as exc_info:
            load_parser("nonexistent")

        assert "Parser module 'nonexistent' not found" in str(exc_info.value)
        assert "Available parsers" in str(exc_info.value)

    def test_parser_returns_correct_signature(self):
        """Test that loaded parser has correct signature."""
        parser = load_parser("thomas")

        # Should return tuple[Optional[int], list[str]]
        result = parser.parse('{"O": 1}')
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], int) or result[0] is None
        assert isinstance(result[1], list)
