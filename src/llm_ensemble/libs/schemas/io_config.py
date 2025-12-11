"""I/O configuration schema.

Defines the Pydantic schema for I/O format configurations that bundle
reader and writer adapters together (e.g., json, parquet).

This is a shared schema used across all CLIs.

Provides convenience methods for instantiating adapters from module paths.
"""

from __future__ import annotations
from importlib import import_module
from typing import Any
from pydantic import ConfigDict, Field

from llm_ensemble.libs.schemas.base_config import BaseConfig
from llm_ensemble.libs.runtime.path_manager import PathManager


class IOConfig(BaseConfig):
    """Domain model for I/O format configuration (mirrors configs/io/*.yaml).

    This is the base I/O configuration schema shared across all CLIs.
    CLI-specific I/O configs can extend this class to add additional fields.

    Provides convenience methods for instantiating reader and writer adapters
    from their module paths, enforcing the dynamic import pattern.
    """

    description: str = Field(description="Human-readable description of the format")
    reader_module: str = Field(description="Full Python module path to reader (e.g., 'llm_ensemble.infer.adapters.io.fully_populated_json_reader')")
    reader_class: str = Field(description="Reader class name in UpperCamelCase (e.g., 'FullyPopulatedJsonReader')")
    writer_module: str = Field(description="Full Python module path to writer (e.g., 'llm_ensemble.infer.adapters.io.fully_populated_json_writer')")
    writer_class: str = Field(description="Writer class name in UpperCamelCase (e.g., 'FullyPopulatedJsonWriter')")

    model_config = ConfigDict(extra="forbid")

    def get_reader(self) -> Any:
        """Instantiate and return the reader adapter.

        Dynamically imports the reader module and instantiates the reader class.

        Returns:
            Instance of the reader adapter

        Raises:
            ImportError: If the reader module cannot be imported
            AttributeError: If the reader class doesn't exist in the module
        """
        return self._instantiate_adapter(self.reader_module, self.reader_class)

    def get_writer(self) -> Any:
        """Instantiate and return the writer adapter.

        Dynamically imports the writer module and instantiates the writer class.

        Returns:
            Instance of the writer adapter

        Raises:
            ImportError: If the writer module cannot be imported
            AttributeError: If the writer class doesn't exist in the module
        """
        return self._instantiate_adapter(self.writer_module, self.writer_class)

    def _instantiate_adapter(self, module_path: str, class_name: str, **kwargs) -> Any:
        """Dynamically instantiate an adapter class from module path and class name.

        Args:
            module_path: Full Python module path (e.g., 'llm_ensemble.infer.adapters.io.json_reader')
            class_name: Class name in UpperCamelCase (e.g., 'JsonReader')
            **kwargs: Additional arguments to pass to the class constructor

        Returns:
            Instance of the adapter class

        Raises:
            ImportError: If the module cannot be imported
            AttributeError: If the class doesn't exist in the module
        """
        try:
            module = import_module(module_path)
            adapter_class = getattr(module, class_name)
            return adapter_class(**kwargs) if kwargs else adapter_class()
        except ImportError as e:
            raise ImportError(f"Failed to import module '{module_path}': {e}") from e
        except AttributeError as e:
            raise AttributeError(f"Class '{class_name}' not found in module '{module_path}': {e}") from e

    @classmethod
    def load(cls, io_format: str, cli_name: str) -> "IOConfig":
        """Load an I/O configuration from YAML file.

        Args:
            io_format: I/O format identifier (e.g., "json", "llm_judge_json")
            cli_name: CLI name (e.g., "ingest", "infer", "aggregate", "evaluate")

        Returns:
            IOConfig object with reader and writer adapter specifications

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If YAML is invalid or missing required fields
        """
        return super().load(
            config_name=io_format,
            config_dir=PathManager.get_io_configs_dir(cli_name)
        )
