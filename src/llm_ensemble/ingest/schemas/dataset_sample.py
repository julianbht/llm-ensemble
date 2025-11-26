"""DatasetSample - a judging sample in a specific dataset context.

This represents a JudgingSample within a NormalizedDataset, tracking its position.
The same JudgingSample can appear in multiple datasets at different positions,
so DatasetSample captures this specific instance.

This is a domain entity that bridges INGEST and INFER:
- INGEST creates DatasetSample entities when building NormalizedDataset
- INFER references DatasetSample.id when recording which sample was judged

Design:
- Natural key: (normalized_dataset_id, judging_sample_id)
- Includes sequence_number for deterministic slicing
- Embeds the JudgingSample for convenient access
"""

from __future__ import annotations
from uuid import UUID
from pydantic import BaseModel, Field

from llm_ensemble.ingest.schemas.judging_sample import JudgingSample
from llm_ensemble.libs.db import compute_dataset_sample_uuid


class DatasetSample(BaseModel):
    """A judging sample in a specific dataset context.

    Links a JudgingSample to a NormalizedDataset with its position.
    This enables tracking which specific dataset sample was judged,
    not just which abstract judging sample.

    The id field is a deterministic UUID computed from
    (normalized_dataset_id, judging_sample.id).
    """

    id: UUID = Field(
        ...,
        description="Deterministic UUID computed from (normalized_dataset_id, judging_sample_id)"
    )

    normalized_dataset_id: UUID = Field(
        ...,
        description="ID of the dataset this sample belongs to"
    )

    judging_sample: JudgingSample = Field(
        ...,
        description="The judging sample content (query + document + gold score)"
    )

    sequence_number: int = Field(
        ...,
        ge=0,
        description="Position of this sample in the dataset (0-indexed, for slicing)"
    )

    @classmethod
    def create(
        cls,
        normalized_dataset_id: UUID,
        judging_sample: JudgingSample,
        sequence_number: int,
    ) -> "DatasetSample":
        """Create a DatasetSample with computed deterministic UUID.

        Args:
            normalized_dataset_id: ID of the dataset this sample belongs to
            judging_sample: The judging sample content
            sequence_number: Position in the dataset (0-indexed)

        Returns:
            DatasetSample instance with computed id
        """
        ds_id = compute_dataset_sample_uuid(
            normalized_dataset_id,
            judging_sample.id
        )
        return cls(
            id=ds_id,
            normalized_dataset_id=normalized_dataset_id,
            judging_sample=judging_sample,
            sequence_number=sequence_number,
        )
