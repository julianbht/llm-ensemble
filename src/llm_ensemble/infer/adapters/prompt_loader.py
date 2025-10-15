"""Prompt template loader for LLM inference.

Loads Jinja2 prompt templates and their YAML configs from the centralized
configs/prompts directory. Follows the same pattern as config_loader.py for consistency.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import yaml
from jinja2 import Template

from llm_ensemble.infer.domain.models import PromptConfig


def get_default_prompts_dir() -> Path:
    """Get the default configs/prompts directory.

    Returns:
        Path to configs/prompts relative to project root
    """
    # Navigate from this file to project root, then to configs/prompts
    # This file is at: src/llm_ensemble/infer/adapters/prompt_loader.py
    # Project root is 4 levels up
    project_root = Path(__file__).parents[4]
    return project_root / "configs" / "prompts"


def load_prompt_config(
    prompt_name: str,
    prompts_dir: Optional[Path] = None,
) -> PromptConfig:
    """Load a prompt configuration from YAML file.

    Args:
        prompt_name: Prompt identifier (e.g., "thomas-et-al-prompt")
        prompts_dir: Directory containing prompt configs (defaults to configs/prompts)

    Returns:
        PromptConfig object with all settings loaded from YAML

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If YAML is invalid or missing required fields

    Example:
        >>> config = load_prompt_config("thomas-et-al-prompt")
        >>> config.variables
        {'role': True, 'aspects': False}
    """
    # Determine prompts directory
    if prompts_dir is None:
        prompts_dir = get_default_prompts_dir()

    # Build path to config file
    config_path = prompts_dir / f"{prompt_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Prompt config not found: {config_path}\n"
            f"Available prompts in {prompts_dir}:\n"
            + "\n".join(f"  - {p.stem}" for p in prompts_dir.glob("*.yaml"))
        )

    # Load YAML
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid config file {config_path}: expected YAML object")

    # Validate and parse into PromptConfig
    try:
        return PromptConfig(**data)
    except Exception as e:
        raise ValueError(f"Failed to parse prompt config {config_path}: {e}") from e


def load_prompt_template(
    template_name: str,
    prompts_dir: Optional[Path] = None,
) -> Template:
    """Load a Jinja2 prompt template from the prompts directory.

    Args:
        template_name: Template filename (with or without .jinja extension)
        prompts_dir: Directory containing prompt templates (defaults to configs/prompts)

    Returns:
        Jinja2 Template object ready for rendering

    Raises:
        FileNotFoundError: If template file doesn't exist

    Example:
        >>> template = load_prompt_template("thomas-et-al-prompt")
        >>> instruction = template.render(query="python", page_text="...")
    """
    # Determine prompts directory
    if prompts_dir is None:
        prompts_dir = get_default_prompts_dir()

    # Add .jinja extension if not present
    if not template_name.endswith(".jinja"):
        template_name = f"{template_name}.jinja"

    # Build path to template file
    template_path = prompts_dir / template_name

    if not template_path.exists():
        raise FileNotFoundError(
            f"Prompt template not found: {template_path}\n"
            f"Available templates in {prompts_dir}:\n"
            + "\n".join(f"  - {p.stem}" for p in prompts_dir.glob("*.jinja"))
        )

    # Load template
    with open(template_path, "r", encoding="utf-8") as f:
        return Template(f.read())
