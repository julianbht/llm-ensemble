"""Shared data models for the infer CLI.

This module contains the canonical ModelJudgement schema that all features
must produce. This is the core domain model for LLM judge outputs.
"""

from __future__ import annotations
from typing import Optional, Literal
from pydantic import BaseModel, Field


class ModelJudgement(BaseModel):
    """Output from a single LLM judge for one query-document pair.

    This is the canonical judgement schema that all adapters must produce.
    Follows the data contract specified in CLAUDE.md.
    """

    # Identity
    model_id: str = Field(..., description="Model identifier (e.g., 'phi3-mini')")
    provider: str = Field(..., description="Provider name (e.g., 'hf', 'ollama')")
    version: Optional[str] = Field(None, description="Model version if available")

    # Input identifiers (from JudgingExample)
    query_id: str = Field(..., description="Query ID from input")
    docid: str = Field(..., description="Document ID from input")

    # Core judgement output
    label: Optional[Literal[0, 1, 2]] = Field(
        None,
        description=(
            "Relevance label: "
            "2 = highly relevant, very helpful for this query; "
            "1 = relevant, may be partly helpful; "
            "0 = not relevant, should never be shown for this query. "
            "None = parsing failed or model did not produce valid output."
        )
    )
    score: Optional[float] = Field(None, ge=0.0, le=2.0, description="Relevance score [0-2]")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Model self-reported or derived uncertainty")

    # Explainability
    rationale: Optional[str] = Field(None, description="Model's explanation for its judgement")
    raw_text: str = Field(..., description="Unparsed model output for debugging")

    # Observability
    latency_ms: float = Field(..., description="Inference time in milliseconds")
    retries: int = Field(0, description="Number of retries attempted")
    cost_estimate: Optional[float] = Field(None, description="Estimated cost in USD")

    # Warnings/metadata
    warnings: list[str] = Field(default_factory=list, description="Parser warnings, fallbacks, etc.")
