"""Prompt builders for different prompt templates.

Each builder module should export a `build()` function that takes:
- template: Jinja2 Template object
- example: Dict with judging example data

And returns a rendered prompt string.
"""

from __future__ import annotations
import importlib
from pathlib import Path
from jinja2 import Template


def load_builder(builder_name: str):
    """Load a prompt builder module by name.

    Args:
        builder_name: Name of the builder module (e.g., "thomas")

    Returns:
        The builder module with a `build()` function

    Raises:
        ImportError: If the builder module doesn't exist
        AttributeError: If the module doesn't have a `build()` function

    Example:
        >>> builder = load_builder("thomas")
        >>> prompt = builder.build(template, example)
    """
    try:
        module = importlib.import_module(f"llm_ensemble.infer.prompts.builders.{builder_name}")
    except ImportError as e:
        # List available builders
        builders_dir = Path(__file__).parent
        available = [
            p.stem for p in builders_dir.glob("*.py")
            if p.stem != "__init__"
        ]
        raise ImportError(
            f"Builder module '{builder_name}' not found. "
            f"Available builders: {', '.join(available)}"
        ) from e

    if not hasattr(module, "build"):
        raise AttributeError(
            f"Builder module '{builder_name}' must define a build() function"
        )

    return module
