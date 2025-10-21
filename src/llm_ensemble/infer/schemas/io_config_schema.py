"""I/O configuration schema.

Defines the Pydantic schema for I/O format configurations that bundle
reader and writer adapters together (e.g., ndjson, parquet).
"""

from __future__ import annotations
from pydantic import BaseModel, Field


class IOConfig(BaseModel):
    """Domain model for I/O format configuration (mirrors configs/io/*.yaml)."""

    io_format: str = Field(description="I/O format identifier (e.g., 'ndjson', 'parquet')")
    description: str = Field(description="Human-readable description of the format")
    reader: str = Field(description="Reader adapter module name")
    writer: str = Field(description="Writer adapter module name")

    class Config:
        """Pydantic config."""

        extra = "forbid"  # Raise error on unexpected fields
