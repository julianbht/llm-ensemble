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

    The name field is injected by config loaders from the filename (without extension).
    The name_hint field allows configs to explicitly contribute to human-readable
    run IDs, making it easy to identify what configurations were used in a run.

    Also provides helper methods for dynamic adapter instantiation from module paths.
    """

    name: Optional[str] = Field(
        None,
        description=(
            "Configuration name, derived from filename (without extension). "
            "This field is injected by config loaders and should NOT be set in YAML files. "
            "Used for identity/UUID generation. Example: 'gpt-oss-20b' from 'gpt-oss-20b.yaml'"
        ),
    )

    name_hint: Optional[str] = Field(
        None,
        description=(
            "Short name hint for run ID generation (e.g., 'gpt20b', 'thomas', 'llmjudge'). "
            "If not provided, this config won't contribute to the run ID. "
            "Keep it short (5-15 chars) and use only alphanumeric characters, hyphens, or underscores."
        ),
    )

    def _instantiate_adapter(self, module_path: str, class_name: str, **kwargs) -> Any:
        """Dynamically instantiate an adapter class from separate module path and class name.

        This method provides a standardized way to instantiate adapter classes
        with clear separation between module path (snake_case) and class name (UpperCamelCase).

        Args:
            module_path: Full Python module path in snake_case
                        (e.g., 'llm_ensemble.infer.adapters.io.fully_populated_json_reader')
            class_name: Class name in UpperCamelCase (e.g., 'FullyPopulatedJsonReader')
            **kwargs: Additional arguments to pass to the class constructor

        Returns:
            Instance of the adapter class

        Raises:
            ImportError: If the module cannot be imported
            AttributeError: If the class doesn't exist in the module

        Example:
            >>> config = IOConfig(...)
            >>> reader = config._instantiate_adapter(
            ...     'llm_ensemble.infer.adapters.io.fully_populated_json_reader',
            ...     'FullyPopulatedJsonReader'
            ... )
        """
        try:
            module = import_module(module_path)
            adapter_class = getattr(module, class_name)
            return adapter_class(**kwargs) if kwargs else adapter_class()
        except ImportError as e:
            raise ImportError(
                f"Failed to import module '{module_path}': {e}"
            ) from e
        except AttributeError as e:
            raise AttributeError(
                f"Class '{class_name}' not found in module '{module_path}': {e}"
            ) from e
