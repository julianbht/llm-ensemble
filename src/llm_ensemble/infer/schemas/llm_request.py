"""LLMRequest - What was sent to the LLM (prompt + warnings from prompt building).

This represents the input sent to the LLM provider during inference.
It captures the rendered prompt text and any warnings that occurred during prompt construction.
"""

from __future__ import annotations
from pydantic import BaseModel, Field

from llm_ensemble.infer.schemas.warnings import BaseWarning


class LLMRequest(BaseModel):
    """Request sent to LLM provider for inference.

    This represents what was sent to the LLM:
    - prompt: The rendered prompt text after template substitution
    - warnings: Issues encountered during prompt building (missing variables, rendering errors, etc.)
    The PromptBuilder adapter creates this by rendering templates with sample data.
    """

    prompt: str = Field(
        ...,
        description="The rendered prompt text sent to the LLM for this inference"
    )

    warnings: list[BaseWarning] = Field(
        default_factory=list,
        description="Prompt builder warnings: rendering errors, missing variables, validation issues, etc."
    )
