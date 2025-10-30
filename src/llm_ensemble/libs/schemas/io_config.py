"""I/O configuration schema.

Defines the Pydantic schema for I/O format configurations that bundle
reader and writer adapters together (e.g., ndjson, parquet).

This is a shared schema used across all CLIs.

Provides convenience methods for instantiating adapters from module paths.
"""

from __future__ import annotations
from typing import Any
from pydantic import Field

from llm_ensemble.libs.schemas.base_config import BaseConfig


class IOConfig(BaseConfig):
    """Domain model for I/O format configuration (mirrors configs/io/*.yaml).

    This is the base I/O configuration schema shared across all CLIs.
    CLI-specific I/O configs can extend this class to add additional fields.

    Provides convenience methods for instantiating reader and writer adapters
    from their module paths, enforcing the dynamic import pattern.
    """

    description: str = Field(description="Human-readable description of the format")
    reader: str = Field(description="Reader adapter module name")
    reader_module_path: str = Field(description="Full module path to reader adapter (e.g., 'llm_ensemble.infer.adapters.io.ndjson_example_reader.NdjsonExampleReader')")
    writer: str = Field(description="Writer adapter module name")
    writer_module_path: str = Field(description="Full module path to writer adapter (e.g., 'llm_ensemble.infer.adapters.io.ndjson_judgement_writer.NdjsonJudgementWriter')")

    class Config:
        """Pydantic config."""

        extra = "forbid"  # Raise error on unexpected fields

    def get_reader(self) -> Any:
        """Instantiate and return the reader adapter.

        Dynamically imports and instantiates the reader class specified
        by reader_module_path.

        Returns:
            Instance of the reader adapter

        Raises:
            ImportError: If the reader module path cannot be imported

        Example:
            >>> config = IOConfig(...)
            >>> reader = config.get_reader()
        """
        return self.instantiate_from_module_path(self.reader_module_path)

    def get_writer(self) -> Any:
        """Instantiate and return the writer adapter.

        Dynamically imports and instantiates the writer class specified
        by writer_module_path.

        Returns:
            Instance of the writer adapter

        Raises:
            ImportError: If the writer module path cannot be imported

        Example:
            >>> config = IOConfig(...)
            >>> writer = config.get_writer()
        """
        return self.instantiate_from_module_path(self.writer_module_path)
