"""IngestManifest schema - extends base Manifest with ingest-specific execution parameters."""

from __future__ import annotations
from typing import Optional
from pydantic import Field

from llm_ensemble.libs.runtime.manifest import Manifest
from llm_ensemble.ingest.schemas.ingest_io_config import IngestIOConfig


class IngestManifest(Manifest):
    """Manifest for ingest CLI runs.

    Extends the base Manifest with ingest-specific execution parameters:
    what the user requested and what configs were used.
    """

    io_config_name: str = Field(
        ...,
        description="Name of the I/O config used (e.g., 'llm_judge_challenge')"
    )

    io_config: IngestIOConfig = Field(
        ...,
        description="I/O configuration used for this run"
    )

    input_path: str = Field(
        ...,
        description="Path to input directory containing raw dataset files"
    )

    limit: Optional[int] = Field(
        default=None,
        description="Maximum number of samples to process (None = no limit)"
    )

    sample_count: Optional[int] = Field(
        default=None,
        description="Number of judging samples produced (set at end of run)"
    )
