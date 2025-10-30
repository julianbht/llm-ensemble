"""Base configuration schema.

Defines the base Pydantic schema for all configuration types that participate
in run ID generation. This ensures consistency across model, prompt, I/O, and
ensemble configurations.

Provides utility methods for dynamic adapter instantiation from module paths.
"""

from __future__ import annotations
from importlib import import_module
from typing import Optional, Any
from pydantic import BaseModel, Field


class BaseConfig(BaseModel):
    """Base configuration class for all configs that participate in run ID generation.

    All configuration types (ModelConfig, PromptConfig, IOConfig, EnsembleConfig)
    should inherit from this base class to ensure consistent metadata fields.

    The name_hint field allows configs to explicitly contribute to human-readable
    run IDs, making it easy to identify what configurations were used in a run.

    Also provides helper methods for dynamic adapter instantiation from module paths.
    """

    name_hint: Optional[str] = Field(
        None,
        description=(
            "Short name hint for run ID generation (e.g., 'gpt20b', 'thomas', 'llmjudge'). "
            "If not provided, this config won't contribute to the run ID. "
            "Keep it short (5-15 chars) and use only alphanumeric characters, hyphens, or underscores."
        ),
    )

    def instantiate_from_module_path(self, module_path: str) -> Any:
        """Dynamically instantiate a class from a module path.

        This method provides a standardized way to instantiate adapter classes
        from fully qualified module paths (e.g., 'pkg.module.ClassName').

        Args:
            module_path: Full module path including class name
                        (e.g., 'llm_ensemble.infer.adapters.io.ndjson_example_reader.NdjsonExampleReader')

        Returns:
            Instance of the class specified by module_path

        Raises:
            ImportError: If the module path cannot be imported or class cannot be found

        Example:
            >>> config = IOConfig(...)
            >>> reader = config.instantiate_from_module_path(config.reader_module_path)
        """
        try:
            module_path_str, class_name = module_path.rsplit(".", 1)
            module = import_module(module_path_str)
            adapter_class = getattr(module, class_name)
            return adapter_class()
        except (ValueError, ImportError, AttributeError) as e:
            raise ImportError(
                f"Failed to instantiate class from '{module_path}': {e}"
            ) from e
