"""I/O configuration schema.

Defines the Pydantic schema for I/O format configurations that bundle
reader and writer adapters together (e.g., json, parquet).

This is a shared schema used across all CLIs.

Provides convenience methods for instantiating adapters from module paths.
"""

from __future__ import annotations
from typing import Any
from pydantic import ConfigDict, Field

from llm_ensemble.libs.schemas.base_config import BaseConfig


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

        Example:
            >>> config = IOConfig(...)
            >>> reader = config.get_reader()
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

        Example:
            >>> config = IOConfig(...)
            >>> writer = config.get_writer()
        """
        return self._instantiate_adapter(self.writer_module, self.writer_class)
