"""NormalizedDataset - container for judging samples and ingest metadata."""

from __future__ import annotations
from typing import List
from pydantic import BaseModel, Field

from llm_ensemble.ingest.schemas.judging_sample import JudgingSample
from llm_ensemble.ingest.schemas.ingest_manifest import IngestManifest


class NormalizedDataset(BaseModel):
    """Container for normalized IR dataset with full provenance.

    Bundles the judging samples with their complete execution manifest,
    ensuring full traceability from raw data to normalized output.
    """

    judging_samples: List[JudgingSample] = Field(
        ...,
        description="List of normalized query-document-relevance samples"
    )

    manifest: IngestManifest = Field(
        ...,
        description="Complete execution manifest: runtime metadata + execution parameters"
    )

    @property
    def sample_count(self) -> int:
        """Get the number of samples in this dataset.

        Returns:
            Count of judging samples
        """
        return len(self.judging_samples)
