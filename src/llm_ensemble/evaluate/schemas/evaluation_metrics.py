"""EvaluationMetrics schema - output from the evaluate CLI.

This is the canonical schema for evaluation metrics output, defining the
final output of the LLM ensemble pipeline.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class ClassMetrics(BaseModel):
    """Per-class metrics for a single relevance label."""

    precision: Optional[float] = Field(None, ge=0.0, le=1.0, description="Precision score")
    recall: Optional[float] = Field(None, ge=0.0, le=1.0, description="Recall score")
    f1: Optional[float] = Field(None, ge=0.0, le=1.0, description="F1 score")
    support: Optional[int] = Field(None, ge=0, description="Number of samples in this class")


class EvaluationMetrics(BaseModel):
    """Output from the evaluate CLI.

    This is the canonical schema for evaluation metrics, containing overall
    metrics and per-class breakdowns for relevance judgement evaluation.
    """

    # Overall metrics
    accuracy: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Overall accuracy"
    )
    macro_f1: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Macro-averaged F1 score"
    )
    micro_f1: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Micro-averaged F1 score"
    )

    # Per-class metrics (keyed by label: 0, 1, or 2)
    per_class_metrics: Optional[dict[str, ClassMetrics]] = Field(
        None,
        description="Metrics broken down by class (0, 1, 2)"
    )

    # Confusion matrix (3x3 for labels 0, 1, 2)
    confusion_matrix: Optional[list[list[int]]] = Field(
        None,
        description="3x3 confusion matrix for relevance labels [0, 1, 2]"
    )
