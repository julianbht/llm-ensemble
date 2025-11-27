"""DatasetJudgement - single position in a judged dataset containing judgements from multiple models.

Represents one query-document pair position in the JudgedDataset, with all LLM judgements
for that position (one judgement per model config that was run).

This is similar to DatasetSample in the ingest pipeline.
"""

from __future__ import annotations
from uuid import UUID
from pydantic import BaseModel, Field

from llm_ensemble.infer.schemas.llm_judgement import LLMJudgement


class DatasetJudgement(BaseModel):
    """Single position in a judged dataset with all model judgements for that position.

    Contains:
    - id: Deterministic UUID from (judged_dataset_id, sequence_number)
    - judged_dataset_id: Which judged dataset this belongs to
    - sequence_number: Position in the judged dataset (0-indexed)
    - llm_judgements: All judgements for this position (one per model config)

    This represents one query-document pair with judgements from multiple models.
    Used in aggregation to group judgements by position across different runs.
    """

    id: UUID = Field(
        ...,
        description="Deterministic UUID computed from judged_dataset_id and sequence_number"
    )

    judged_dataset_id: UUID = Field(
        ...,
        description="Which judged dataset this judgement belongs to"
    )

    sequence_number: int = Field(
        ...,
        ge=0,
        description="Position in the judged dataset (0-indexed, preserves order)"
    )

    llm_judgements: list[LLMJudgement] = Field(
        default_factory=list,
        description="All LLM judgements for this position (one per model config)"
    )
