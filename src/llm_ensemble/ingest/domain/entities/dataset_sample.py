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
from uuid import UUID, uuid4
from pydantic import BaseModel, Field

from llm_ensemble.ingest.domain.entities.judging_sample import JudgingSample


class DatasetSample(BaseModel):
    """A judging sample in a specific dataset context.

    Links a JudgingSample to a NormalizedDataset with its position.
    This enables tracking which specific dataset sample was judged,
    not just which abstract judging sample.

    The id field is a random UUID (v4).
    """

    id: UUID = Field(
        default_factory=uuid4,
        description="Random UUID identifier"
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
