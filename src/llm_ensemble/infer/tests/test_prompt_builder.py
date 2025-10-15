"""Tests for prompt builder functionality."""

from pathlib import Path
import pytest
from jinja2 import Template

from llm_ensemble.infer.domain.prompt_builder import (
    build_instruction,
    build_instruction_from_judging_example,
)


# Fixture: Load the thomas-et-al template once for all tests
@pytest.fixture
def thomas_template():
    """Load the thomas-et-al-prompt.jinja template."""
    template_path = Path(__file__).parents[1] / "domain" / "prompts" / "thomas-et-al-prompt.jinja"
    with open(template_path) as f:
        return Template(f.read())


def test_basic_instruction_without_role(thomas_template):
    """Test building instruction without role description."""
    result = build_instruction(
        template=thomas_template,
        query="python tutorial",
        page_text="This page teaches Python basics.",
        role=False,
    )

    assert "python tutorial" in result
    assert "This page teaches Python basics." in result
    assert "You are a search quality rater" not in result
    assert "Given a query and a web page" in result
    assert '{"O":' in result  # Should have simple output format


def test_instruction_with_description_and_narrative(thomas_template):
    """Test including optional description and narrative."""
    result = build_instruction(
        template=thomas_template,
        query="machine learning",
        page_text="ML is a subset of AI.",
        description="tutorials for beginners",
        narrative="The user wants to learn ML from scratch.",
    )

    assert "machine learning" in result
    assert "tutorials for beginners" in result
    assert "The user wants to learn ML from scratch." in result
    assert "They were looking for:" in result


def test_instruction_with_aspects(thomas_template):
    """Test multi-aspect evaluation format (M/T/O)."""
    result = build_instruction(
        template=thomas_template,
        query="best restaurants",
        page_text="Here are the top 10 restaurants...",
        aspects=True,
    )

    assert "best restaurants" in result
    assert "Measure how well the content matches" in result
    assert "Measure how trustworthy the web page is" in result
    assert '"M"' in result
    assert '"O"' in result  # Should have M/T/O format


def test_build_from_judging_example(thomas_template):
    """Test convenience wrapper for JudgingExample dict."""
    example = {
        "query_text": "python vs java",
        "doc": "Python is easier for beginners...",
        "query_id": "q1",
        "docid": "d1",
    }

    result = build_instruction_from_judging_example(
        template=thomas_template,
        example=example,
        role=True,
        description="comparison article",
    )

    assert "python vs java" in result
    assert "Python is easier for beginners" in result
    assert "comparison article" in result
    assert "You are a search quality rater" in result


def test_minimal_instruction(thomas_template):
    """Test minimal instruction with only required fields."""
    result = build_instruction(
        template=thomas_template,
        query="test",
        page_text="content",
    )

    # Should have all core components
    assert "test" in result
    assert "content" in result
    assert "0 to 2" in result  # Score scale
    assert "—BEGIN WEB PAGE CONTENT—" in result
    assert "—END WEB PAGE CONTENT—" in result
