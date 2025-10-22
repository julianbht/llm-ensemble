"""EnsembleResult schema - output from the aggregate CLI.

This is the canonical schema for ensemble aggregation output, defining the
data contract between the aggregate and evaluate CLIs.
"""

from __future__ import annotations
from typing import Optional, Literal
from pydantic import BaseModel, Field


class ModelVote(BaseModel):
    """Individual model vote used in ensemble aggregation."""

    model_id: str = Field(..., description="Model identifier")
    label: Optional[Literal[0, 1, 2]] = Field(None, description="Model's predicted label")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Model's confidence score")


class EnsembleResult(BaseModel):
    """Output from ensemble aggregation for one query-document pair.

    This is the canonical schema that the aggregate CLI produces and the
    evaluate CLI consumes.
    """

    # Identity (from input JudgingExample)
    query_id: str = Field(..., description="Query ID")
    docid: str = Field(..., description="Document ID")

    # Aggregated decision
    final_label: Literal[0, 1, 2] = Field(
        ...,
        description=(
            "Ensemble aggregated label: "
            "2 = highly relevant, "
            "1 = relevant, "
            "0 = not relevant"
        )
    )
    final_confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Aggregated confidence score"
    )

    # Explainability
    per_model_votes: Optional[list[ModelVote]] = Field(
        None,
        description="Individual model votes used in aggregation"
    )
    aggregation_strategy: Optional[str] = Field(
        None,
        description="Name of aggregation strategy used (e.g., 'weighted_majority', 'unanimous')"
    )

    # Warnings/metadata
    warnings: list[str] = Field(
        default_factory=list,
        description="Warnings from aggregation (e.g., ties, low agreement)"
    )
