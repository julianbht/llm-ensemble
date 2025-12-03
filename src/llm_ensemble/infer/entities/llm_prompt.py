"""LLMPrompt entity for the infer CLI.

Rendered prompt text sent to LLM, built from a dataset sample.
"""

from __future__ import annotations
import uuid
from uuid import UUID
from pydantic import BaseModel, Field

from llm_ensemble.ingest.schemas.dataset_sample import DatasetSample


class LLMPrompt(BaseModel):
    """Rendered prompt text sent to LLM, built from a dataset sample.

    Captures the semantic relationship: prompt is built FROM dataset_sample.
    """

    id: UUID = Field(
        default_factory=uuid.uuid4,
        description="Random UUID for this prompt"
    )

    dataset_sample: DatasetSample = Field(
        ...,
        description="The dataset sample this prompt was built from"
    )

    prompt_text: str = Field(
        ...,
        description="The rendered prompt text sent to the LLM"
    )
