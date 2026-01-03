"""NormalizedDataset - pure data entity for internal dataset representation.

Represents a specific collection of judging samples with a content-based fingerprint.
Multiple ingest runs with identical data produce the same fingerprint, enabling
deduplication and idempotent re-runs.

Design:
- Pure data carrier - no business logic or factory methods
- Fingerprint computed by domain builder (normalized_dataset_builder.py)
- All samples from one external dataset
- Samples stored in deterministic order for reproducibility
"""

from __future__ import annotations
from uuid import UUID, uuid4
from typing import Optional
from pydantic import BaseModel, Field

from llm_ensemble.ingest.domain.entities.dataset_sample import NormalizedDatasetJudgingSample


class NormalizedDataset(BaseModel):
    """Pure data entity representing an internal dataset.

    Contains a collection of judging samples with a content-based fingerprint for
    deduplication. Same data across runs produces the same fingerprint, enabling:
    - Idempotent re-runs (no duplicate datasets)
    - Database constraint enforcement (unique fingerprint)
    - Efficient dataset comparison

    Samples are wrapped in DatasetSample objects with sequence numbers and stored
    in deterministic order for reproducibility.

    Use `build_normalized_dataset()` from normalized_dataset_builder.py to create instances.
    """

    id: UUID = Field(
        default_factory=uuid4,
        description="Random UUID identifier"
    )
    fingerprint: str = Field(
        ...,
        description="SHA256 hash of sample content (query/doc hashes + scores) for deduplication"
    )
    external_dataset_name: Optional[str] = Field(
        None,
        description="Name of the external source dataset (e.g., 'msmarco', 'llmjudge')"
    )
    samples: list[NormalizedDatasetJudgingSample] = Field(
        ...,
        description="Dataset samples with sequence numbers, sorted by content for reproducibility"
    )

    @property
    def sample_count(self) -> int:
        """Get number of samples in this dataset."""
        return len(self.samples)
