"""AggregateRunInfo schema - extends base RunInfo with aggregate-specific configuration.

This contains all aggregate-specific configuration that is known before the run
starts and remains immutable throughout execution.
"""

from __future__ import annotations
from pydantic import Field

from llm_ensemble.libs.runtime.run_info import RunInfo
from llm_ensemble.libs.schemas import IOConfig
from llm_ensemble.aggregate.schemas.ensemble_config_schema import EnsembleConfig


class AggregateRunInfo(RunInfo):
    """Runtime context for aggregate CLI runs.
    
    Extends the base RunInfo with aggregate-specific configuration metadata:
    - Which configs were used (ensemble, I/O)
    - Full configuration objects for reproducibility
    - Input parameters (file paths from infer runs)
    
    All fields in this class are immutable and known before processing begins,
    allowing AggregatedJudgement objects to embed complete provenance as soon as
    they are created, without waiting for aggregate statistics.
    
    This is separate from AggregateRunSummary which contains post-run metrics like
    judgement counts, tie statistics, and warnings summary.
    """
    
    # Override cli_name from base RunInfo to automatically set it to "aggregate"
    cli_name: str = Field(
        default="aggregate",
        description="Name of the CLI that generated this run (always 'aggregate' for AggregateRunInfo)"
    )
    
    # Configuration names (what user requested)
    ensemble_config_name: str = Field(
        ...,
        description="Name of the ensemble config used (e.g., 'weighted_majority_v1')"
    )
    
    io_config_name: str = Field(
        ...,
        description="Name of the I/O config used (e.g., 'json')"
    )
    
    # Full configuration objects (for reproducibility)
    ensemble_config: EnsembleConfig = Field(
        ...,
        description="Ensemble configuration used for this run (strategy and parameters)"
    )
    
    io_config: IOConfig = Field(
        ...,
        description="I/O configuration used for this run"
    )
    
    # Input parameters
    input_run_names: list[str] = Field(
        ...,
        description="List of infer run identifiers to read judgements from (e.g., ['run1', 'run2'])"
    )
    
    # Pydantic-specific pattern to make this class immutable
    class Config:
        """Pydantic config."""
        frozen = True  # Make immutable to emphasize this is runtime context
