"""Ingest I/O configuration schema.

Extends the base IOConfig with ingest-specific fields like dataset_id and data_dir.
"""

from __future__ import annotations
from pathlib import Path
from pydantic import Field

from llm_ensemble.infer.schemas import IOConfig


class IngestIOConfig(IOConfig):
    """I/O configuration for ingest CLI.

    Extends IOConfig with ingest-specific fields:
    - dataset_id: Identifier embedded in JudgingExample records
    - data_dir: Default directory containing raw dataset files
    """

    dataset_id: str = Field(description="Dataset identifier for JudgingExample records")
    data_dir: Path = Field(description="Default data directory containing raw dataset files")
