"""LLMScore entity for the infer CLI.

Parsed relevance assessment extracted from LLM response.
Mirrors LLMScoreORM structure - just the parsed fields.
"""

from __future__ import annotations
import uuid
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field

from llm_ensemble.libs.schemas.relevance_score import RelevanceScore


class LLMScore(BaseModel):
    """Parsed relevance assessment extracted from LLM response.

    This represents the structured score that a ResponseParser extracts
    from raw LLM output text. All fields are optional to handle parse failures.

    Parser warnings are NOT stored here - they belong on LLMJudgement since
    they're about the judgement process (not the score result).

    The parser used and the response text are also not stored here - they are:
    - parser: stored in AdapterConfig (at JudgedDataset level)
    - response_text: stored directly on LLMJudgement
    - warnings: stored on LLMJudgement
    """

    id: UUID = Field(
        default_factory=uuid.uuid4,
        description="Random UUID for this score"
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
