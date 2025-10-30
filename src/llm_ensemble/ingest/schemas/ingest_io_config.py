"""Ingest I/O configuration schema.

Extends the base IOConfig with ingest-specific fields like dataset_id.
"""

from __future__ import annotations
from pydantic import Field

from llm_ensemble.libs.schemas import IOConfig


class IngestIOConfig(IOConfig):
    """I/O configuration for ingest CLI.

    Extends IOConfig with ingest-specific fields:
    - dataset_id: Identifier embedded in JudgingExample records

    Note: Input path is provided via --input CLI flag, not in config.
    """

    dataset_id: str = Field(description="Dataset identifier for JudgingExample records")
