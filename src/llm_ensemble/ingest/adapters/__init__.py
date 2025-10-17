from __future__ import annotations

import importlib
from pathlib import Path
from typing import Callable, Generator

from llm_ensemble.ingest.schemas import JudgingExample


# Type alias for adapter functions
AdapterFunction = Callable[[Path], Generator[JudgingExample, None, None]]


def load_adapter(adapter_name: str) -> AdapterFunction:
    """Dynamically load an ingest adapter by name.

    Args:
        adapter_name: Name of the adapter module (e.g., 'llm_judge')

    Returns:
        The iter_examples function from the adapter module

    Raises:
        ImportError: If adapter module doesn't exist
        AttributeError: If adapter doesn't have iter_examples function
    """
    module_path = f"llm_ensemble.ingest.adapters.{adapter_name}"

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise ImportError(f"Adapter '{adapter_name}' not found. Expected module: {module_path}") from e

    if not hasattr(module, "iter_examples"):
        raise AttributeError(
            f"Adapter '{adapter_name}' must provide an 'iter_examples' function "
            f"with signature: iter_examples(base_dir: Path) -> Generator[JudgingExample, None, None]"
        )

    return module.iter_examples


__all__ = ["load_adapter", "AdapterFunction"]
