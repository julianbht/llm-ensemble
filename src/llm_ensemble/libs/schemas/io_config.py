"""I/O configuration schema.

Defines the Pydantic schema for I/O format configurations that bundle
reader and writer adapters together (e.g., ndjson, parquet).

This is a shared schema used across all CLIs.
"""

from __future__ import annotations
from pydantic import Field

from llm_ensemble.libs.schemas.base_config import BaseConfig


class IOConfig(BaseConfig):
    """Domain model for I/O format configuration (mirrors configs/io/*.yaml).

    This is the base I/O configuration schema shared across all CLIs.
    CLI-specific I/O configs can extend this class to add additional fields.
    """

    io_format: str = Field(description="I/O format identifier (e.g., 'ndjson', 'parquet')")
    description: str = Field(description="Human-readable description of the format")
    reader: str = Field(description="Reader adapter module name")
    reader_module_path: str = Field(description="Full module path to reader adapter (e.g., 'llm_ensemble.infer.adapters.io.ndjson_example_reader.NdjsonExampleReader')")
    writer: str = Field(description="Writer adapter module name")
    writer_module_path: str = Field(description="Full module path to writer adapter (e.g., 'llm_ensemble.infer.adapters.io.ndjson_judgement_writer.NdjsonJudgementWriter')")

    class Config:
        """Pydantic config."""

        extra = "forbid"  # Raise error on unexpected fields
