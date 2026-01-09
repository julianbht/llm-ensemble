"""EvaluateRunConfig - immutable configuration bundle for an evaluate run.

Pure Pydantic model containing all configuration needed to execute evaluation:
- I/O configuration (which I/O adapter to use)
- Input run (which run to evaluate - infer or aggregate)
- Metrics (which metrics were computed)

This is a serializable domain entity (frozen Pydantic model).
Separate from EvaluateRun (git info, timestamps, run metadata).
"""

from __future__ import annotations
from uuid import UUID, uuid4
from pydantic import BaseModel, ConfigDict, Field


class EvaluateRunConfig(BaseModel):
    """Immutable configuration bundle for an evaluate run.

    Pure Pydantic model - no business logic, just data.
    Contains configuration decisions made for this specific evaluation run.
    """

    id: UUID = Field(
        default_factory=uuid4,
        description="Random UUID for this config bundle"
    )

    io_config_name: str = Field(
        ...,
        description="Name of the I/O config used (e.g., 'json', 'db_infer', 'dummy')"
    )

    input_run_name: str = Field(
        ...,
        description="Run identifier to evaluate (infer or aggregate run)"
    )

    metric_names: list[str] = Field(
        ...,
        description="List of metric names computed (e.g., ['cohens_kappa', 'krippendorffs_alpha'])"
    )

    model_config = ConfigDict(frozen=True)
