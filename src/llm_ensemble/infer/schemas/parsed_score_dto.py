"""DTO for parsed score data from response parsers.

Simple data structure returned by parser adapters before domain object creation.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field

from llm_ensemble.infer.schemas.warnings import BaseWarning
from llm_ensemble.libs.schemas import RelevanceScore


class ParsedScoreDTO(BaseModel):
    """Raw parsed score data from response parser adapters.

    Adapters implement parse_raw() and return this DTO.
    The port layer maps this to LLMScore domain objects.
    """

    llm_response_text: str = Field(
        ...,
        description="The raw LLM response text that was parsed"
    )

    label: Optional[RelevanceScore] = Field(
        None,
        description="Parsed relevance label (None if parsing failed)"
    )

    confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence score [0-1] (None if not available)"
    )

    rationale: Optional[str] = Field(
        None,
        description="Explanation for the judgement (None if not available)"
    )

    warnings: list[BaseWarning] = Field(
        default_factory=list,
        description="Parser warnings: parse errors, missing fields, validation issues"
    )
