"""Prompt builders for different prompt templates.

Each builder module should export a concrete class that implements the
PromptBuilder port interface.
"""

from __future__ import annotations
import importlib
from pathlib import Path

from llm_ensemble.infer.ports import PromptBuilder


def load_builder(builder_name: str) -> PromptBuilder:
    """Load a prompt builder adapter by name.

    Args:
        builder_name: Name of the builder module (e.g., "thomas")

    Returns:
        An instance of the PromptBuilder implementation

    Raises:
        ImportError: If the builder module doesn't exist
        AttributeError: If the module doesn't have the expected class

    Example:
        >>> builder = load_builder("thomas")
        >>> prompt = builder.build(template, example)
    """
    try:
        module = importlib.import_module(f"llm_ensemble.infer.prompt_builders.{builder_name}")
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

    # Convert builder_name to class name (e.g., "thomas" -> "ThomasPromptBuilder")
    class_name = "".join(word.capitalize() for word in builder_name.split("_")) + "PromptBuilder"
    
    if not hasattr(module, class_name):
        raise AttributeError(
            f"Builder module '{builder_name}' must define a {class_name} class "
            f"that implements the PromptBuilder interface"
        )

    builder_class = getattr(module, class_name)
    return builder_class()


__all__ = ["load_builder"]
