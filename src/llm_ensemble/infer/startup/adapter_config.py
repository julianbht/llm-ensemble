"""Adapter configuration for inference pipeline.

Startup Layer - Request Objects

Lightweight configuration objects that the CLI builds and passes to the runner.
These define WHAT adapters to use and HOW to execute the inference.

Separation of concerns:
- AdapterConfig: Which adapters to instantiate (config names, not loaded configs)
- ExecutionParams: How to execute the inference (run metadata, execution context)

These are separate from:
- InferRunConfig: Domain entity with loaded configs (for use case provenance)
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class AdapterConfig(BaseModel):
    """Adapter selection configuration.

    Lightweight config specifying WHICH adapters to use (by name).
    Built by the CLI, passed to dependency configurator.

    Contains only config names (strings), not loaded config objects.
    The configurator uses these names to load configs and instantiate adapters.
    """

    model_config_name: str = Field(
        ...,
        description="Name of model config file (e.g., 'gpt-oss-20b')"
    )

    provider_name: str = Field(
        ...,
        description="Provider name (e.g., 'openrouter', 'ollama')"
    )

    prompt_template_name: str = Field(
        ...,
        description="Prompt template name (e.g., 'thomas-simple')"
    )

    retry_config_name: str = Field(
        ...,
        description="Retry config name (e.g., 'standard')"
    )

    io_name: str = Field(
        ...,
        description="I/O adapter name (e.g., 'db_to_json', 'db_to_db')"
    )


class ExecutionParams(BaseModel):
    """Execution parameters for inference run.

    Lightweight config specifying HOW to execute the inference.
    Built by the CLI, passed to runner.

    Contains execution context and run metadata.
    """

    input_run_name: str = Field(
        ...,
        description="Ingest run identifier to read samples from"
    )

    start_idx: Optional[int] = Field(
        default=None,
        description="Start index into NormalizedDataset (None = from beginning)"
    )

    end_idx: Optional[int] = Field(
        default=None,
        description="End index into NormalizedDataset (None = until end)"
    )

    run_name: Optional[str] = Field(
        default=None,
        description="Custom run name (auto-generates if not provided)"
    )

    official: bool = Field(
        default=False,
        description="Mark as official run"
    )

    notes: Optional[str] = Field(
        default=None,
        description="Notes about this run (experiment purpose, hypothesis, etc.)"
    )

    tag: Optional[str] = Field(
        default=None,
        description="Tag name for easy reference by downstream CLIs"
    )
