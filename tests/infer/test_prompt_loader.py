"""Tests for prompt loader functionality."""

import pytest

from llm_ensemble.infer.configs.prompts import load_prompt_config
from llm_ensemble.infer.prompts.templates import load_prompt_template
from llm_ensemble.infer.configs.schemas import PromptConfig


@pytest.mark.unit
def test_load_thomas_et_al_config():
    """Test loading the thomas-et-al-prompt config."""
    config = load_prompt_config("thomas-et-al-prompt")

    assert isinstance(config, PromptConfig)
    assert config.name == "thomas-et-al-prompt"
    assert config.template_file == "thomas-et-al-prompt.jinja"
    assert "role" in config.variables
    assert "aspects" in config.variables
    assert config.variables["role"] is True
    assert config.variables["aspects"] is False
    assert config.expected_output_format == "json"
    assert config.response_parser == "parse_thomas_response"


@pytest.mark.unit
def test_load_relevance_v1_config():
    """Test loading the relevance_v1 config."""
    config = load_prompt_config("relevance_v1")

    assert isinstance(config, PromptConfig)
    assert config.name == "relevance_v1"
    assert config.template_file == "relevance_v1.jinja"
    assert "query" in config.variables
    assert "candidate" in config.variables
    assert config.expected_output_format == "json"


@pytest.mark.unit
def test_load_thomas_et_al_template():
    """Test loading the thomas-et-al-prompt template."""
    template = load_prompt_template("thomas-et-al-prompt.jinja")

    # Should be a Jinja2 template
    assert hasattr(template, "render")

    # Test basic rendering
    result = template.render(
        query="test query",
        page_text="test content",
        role=True,
        aspects=False,
    )

    assert "test query" in result
    assert "test content" in result
    assert "You are a search quality rater" in result


@pytest.mark.unit
def test_load_prompt_config_not_found():
    """Test error handling when config doesn't exist."""
    with pytest.raises(FileNotFoundError) as exc_info:
        load_prompt_config("nonexistent-prompt")

    assert "Prompt config not found" in str(exc_info.value)
    assert "Available prompts" in str(exc_info.value)


@pytest.mark.unit
def test_load_prompt_template_not_found():
    """Test error handling when template doesn't exist."""
    with pytest.raises(FileNotFoundError) as exc_info:
        load_prompt_template("nonexistent-template.jinja")

    assert "Prompt template not found" in str(exc_info.value)
    assert "Available templates" in str(exc_info.value)


@pytest.mark.unit
def test_load_config_with_custom_dir(tmp_path, write_file):
    """Test loading config from a custom directory."""
    # Create a test config file using write_file fixture
    write_file(
        tmp_path,
        "test-prompt.yaml",
        """name: test-prompt
template_file: test-prompt.jinja
description: A test prompt
variables:
  foo: bar
  enabled: true
expected_output_format: text
"""
    )

    # Load it
    config = load_prompt_config("test-prompt", prompts_dir=tmp_path)

    assert config.name == "test-prompt"
    assert config.variables["foo"] == "bar"
    assert config.variables["enabled"] is True
    assert config.expected_output_format == "text"
