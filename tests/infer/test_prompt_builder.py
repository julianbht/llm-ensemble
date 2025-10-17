"""Tests for prompt builder functionality."""

import pytest

from llm_ensemble.infer.prompts.builder import (
    build_instruction,
    build_instruction_from_judging_example,
)


# Note: thomas_prompt_template fixture is defined in conftest.py (session-scoped)


@pytest.mark.unit
def test_basic_instruction_without_role(thomas_prompt_template):
    """Test building instruction without role description."""
    result = build_instruction(
        template=thomas_prompt_template,
        query="python tutorial",
        page_text="This page teaches Python basics.",
        role=False,
    )

    assert "python tutorial" in result
    assert "This page teaches Python basics." in result
    assert "You are a search quality rater" not in result
    assert "Given a query and a web page" in result
    assert '{"O":' in result  # Should have simple output format


@pytest.mark.unit
def test_instruction_with_description_and_narrative(thomas_prompt_template):
    """Test including optional description and narrative."""
    result = build_instruction(
        template=thomas_prompt_template,
        query="machine learning",
        page_text="ML is a subset of AI.",
        description="tutorials for beginners",
        narrative="The user wants to learn ML from scratch.",
    )

    assert "machine learning" in result
    assert "tutorials for beginners" in result
    assert "The user wants to learn ML from scratch." in result
    assert "They were looking for:" in result


@pytest.mark.unit
def test_instruction_with_aspects(thomas_prompt_template):
    """Test multi-aspect evaluation format (M/T/O)."""
    result = build_instruction(
        template=thomas_prompt_template,
        query="best restaurants",
        page_text="Here are the top 10 restaurants...",
        aspects=True,
    )

    assert "best restaurants" in result
    assert "Measure how well the content matches" in result
    assert "Measure how trustworthy the web page is" in result
    assert '"M"' in result
    assert '"O"' in result  # Should have M/T/O format


@pytest.mark.unit
def test_build_from_judging_example(thomas_prompt_template):
    """Test convenience wrapper for JudgingExample dict."""
    example = {
        "query_text": "python vs java",
        "doc": "Python is easier for beginners...",
        "query_id": "q1",
        "docid": "d1",
    }

    result = build_instruction_from_judging_example(
        template=thomas_prompt_template,
        example=example,
        role=True,
        description="comparison article",
    )

    assert "python vs java" in result
    assert "Python is easier for beginners" in result
    assert "comparison article" in result
    assert "You are a search quality rater" in result


@pytest.mark.unit
def test_minimal_instruction(thomas_prompt_template):
    """Test minimal instruction with only required fields."""
    result = build_instruction(
        template=thomas_prompt_template,
        query="test",
        page_text="content",
    )

    # Should have all core components
    assert "test" in result
    assert "content" in result
    assert "0 to 2" in result  # Score scale
    assert "—BEGIN WEB PAGE CONTENT—" in result
    assert "—END WEB PAGE CONTENT—" in result
