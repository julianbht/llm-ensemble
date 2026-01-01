"""MetricResult - standardized metric computation result entity.

Domain entity representing the output of metric computation adapters.
All metric adapters return this standardized format.

For now, only scalar metrics are supported.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class MetricResult(BaseModel):
    """Standardized result from scalar metric computation.

    All metric adapters must return this entity for consistent reporting.
    Currently only supports scalar (single numeric value) metrics.
    """

    name: str = Field(
        ...,
        description="Metric name (e.g., 'cohens_kappa', 'kendalls_tau')"
    )

    value: float = Field(
        ...,
        description="Computed metric value (scalar)"
    )

    sample_size: int = Field(
        ...,
        ge=1,
        description="Number of samples used in computation"
    )

    interpretation: Optional[str] = Field(
        None,
        description="Human-readable interpretation of the value (e.g., 'fair', 'substantial')"
    )

    description: Optional[str] = Field(
        None,
        description="Brief description of what this metric measures"
    )
