"""LLMScore entity for the infer CLI.

Parsed relevance assessment extracted from LLM response.
"""

from __future__ import annotations
import uuid
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field

from llm_ensemble.infer.entities.warnings import BaseWarning
from llm_ensemble.libs.schemas import RelevanceScore


class LLMScore(BaseModel):
    """Parsed relevance assessment extracted from LLM response.

    Captures the semantic relationship: score is parsed FROM llm_response_text.
    This represents the structured score that a ResponseParser extracts
    from raw LLM output text. All fields are optional to handle parse failures.
    """

    id: UUID = Field(
        default_factory=uuid.uuid4,
        description="Random UUID for this score"
    )

    llm_response_text: str = Field(
        ...,
        description="The raw LLM response text that was parsed to extract this score"
    )

    label: Optional[RelevanceScore] = Field(
        None,
        description=(
            "Parsed relevance label: "
            "0 = IRRELEVANT, 1 = RELEVANT, 2 = HIGHLY_RELEVANT, 3 = PERFECTLY_RELEVANT. "
            "None if parsing failed."
        )
    )

    confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="LLM self-reported or derived confidence score [0-1]. None if not available."
    )

    rationale: Optional[str] = Field(
        None,
        description="LLM's explanation for its relevance judgement. None if not parseable."
    )

    warnings: list[BaseWarning] = Field(
        default_factory=list,
        description="Parser-level warnings: parse errors, missing fields, validation issues, etc."
    )
