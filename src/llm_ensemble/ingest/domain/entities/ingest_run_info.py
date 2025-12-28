"""IngestRunInfo schema - aggregate root for ingest CLI runs.

Aggregate root that represents the complete output of an ingest run.
Embeds the NormalizedDataset (which in turn embeds IngestRunConfig).

Aggregate structure:
- IngestRunInfo (root): Run metadata (git SHA, timestamps, run_type, notes)
  └── NormalizedDataset: The output dataset
      └── IngestRunConfig: Configuration used to create this dataset
"""

from __future__ import annotations
from pydantic import ConfigDict, Field

from llm_ensemble.libs.runtime.run_info import RunInfo
from llm_ensemble.ingest.domain.entities.normalized_dataset import NormalizedDataset


class IngestRunInfo(RunInfo):
    """Aggregate root for ingest CLI runs.

    Pure Pydantic model representing the complete output of an ingest run.
    Contains:
    - Run metadata (inherited from RunInfo): id, run_name, git info, timestamps, notes
    - Embedded NormalizedDataset: The output dataset with samples, fingerprint, and run_config

    This is the single aggregate root passed to writers and represents the
    complete execution result.
    """

    cli_name: str = Field(
        default="ingest",
        description="Name of the CLI that generated this run (always 'ingest' for IngestRunInfo)"
    )

    normalized_dataset: NormalizedDataset = Field(
        ...,
        description="The normalized dataset produced by this run (embeds run_config)"
    )

    model_config = ConfigDict(frozen=True)
