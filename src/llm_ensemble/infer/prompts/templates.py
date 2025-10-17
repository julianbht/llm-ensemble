"""Prompt template loader.

Loads Jinja2 prompt templates from the centralized configs/prompts directory.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
from jinja2 import Template


def get_default_prompts_dir() -> Path:
    """Get the default configs/prompts directory.

    Returns:
        Path to configs/prompts relative to project root
    """
    # Navigate from this file to project root, then to configs/prompts
    # This file is at: src/llm_ensemble/infer/prompts/templates.py
    # Project root is 4 levels up
    project_root = Path(__file__).parents[4]
    return project_root / "configs" / "prompts"


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
