"""Tests for prompt loader functionality."""

import pytest

from llm_ensemble.infer.config_loaders.prompts import load_prompt_config
from llm_ensemble.infer.config_loaders.templates import load_prompt_template
from llm_ensemble.infer.schemas.prompt_config import PromptConfig


@pytest.mark.unit
def test_load_thomas_et_al_config():
    """Test loading the thomas-et-al-prompt config."""
    config = load_prompt_config("thomas-et-al-prompt")

    assert isinstance(config, PromptConfig)
    assert config.name == "thomas-et-al-prompt"
    assert config.prompt_template == "thomas-et-al-prompt"
    assert config.prompt_builder == "thomas"
    assert config.response_parser == "thomas"


@pytest.mark.unit
def test_load_thomas_et_al_template():
    """Test loading the thomas-et-al-prompt template."""
    template = load_prompt_template("thomas-et-al-prompt.jinja")

    # Should be a Jinja2 template
    assert hasattr(template, "render")

    # Test basic rendering (template uses query and page_text variables)
    result = template.render(
        query="test query",
        page_text="test content",
    )

    assert "test query" in result
    assert "test content" in result


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
description: A test prompt
prompt_template: test-prompt
prompt_builder: test
response_parser: test
"""
    )

    # Load it
    config = load_prompt_config("test-prompt", prompts_dir=tmp_path)

    assert config.name == "test-prompt"
    assert config.prompt_template == "test-prompt"
    assert config.prompt_builder == "test"
    assert config.response_parser == "test"
