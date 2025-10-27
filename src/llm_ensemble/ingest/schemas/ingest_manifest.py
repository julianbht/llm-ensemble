"""IngestManifest schema - extends base Manifest with ingest-specific execution parameters."""

from __future__ import annotations
from typing import Any, Optional
from pydantic import Field

from llm_ensemble.libs.schemas.manifest import Manifest
from llm_ensemble.ingest.schemas.ingest_io_config import IngestIOConfig


class IngestManifest(Manifest):
    """Manifest for ingest CLI runs.

    Extends the base Manifest with ingest-specific execution parameters:
    what the user requested and what configs were used.
    """

    io_config_name: str = Field(
        ...,
        description="Name of the I/O config used (e.g., 'llm_judge_ingest')"
    )

    io_config: IngestIOConfig = Field(
        ...,
        description="Resolved I/O configuration after applying overrides"
    )

    limit: Optional[int] = Field(
        default=None,
        description="Maximum number of samples to process (None = no limit)"
    )

    config_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Config overrides applied via --override flags"
    )

    sample_count: Optional[int] = Field(
        default=None,
        description="Number of judging samples produced (set at end of run)"
    )
