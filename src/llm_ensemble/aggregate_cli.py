"""Aggregate CLI - Combine judgements using ensemble strategies."""

from __future__ import annotations
import typer

from llm_ensemble.aggregate.orchestrator import run_aggregation
from llm_ensemble.aggregate.config_loaders import load_ensemble_config
from llm_ensemble.libs.config import load_io_config
from llm_ensemble.libs.config.logging_config_loader import load_logging_config
from llm_ensemble.libs.runtime.env import load_runtime_config
from llm_ensemble.libs.utils.config_overrides import parse_and_route_overrides, apply_overrides
from llm_ensemble.libs.cli.params import (
    RunName,
    LogCfg,
    Official,
    Notes,
    Override,
    EnsembleCfg,
    AggregateIoCfg,
    InferRunInput,
    Tag,
)

# Load runtime configuration early
load_runtime_config()

app = typer.Typer(
    add_completion=True,
    help="LLM Ensemble – aggregate CLI",
    pretty_exceptions_enable=False,  # Disable Rich verbose tracebacks
)


@app.command("aggregate")
def aggregate(
    # Required parameters
    ensemble_cfg: EnsembleCfg,
    io_cfg: AggregateIoCfg,
    input_run_names: InferRunInput,
    # Optional parameters
    run_name: RunName = None,
    log_cfg: LogCfg = "observability",
    official: Official = False,
    notes: Notes = None,
    override: Override = [],
    tag: Tag = None,
):
    """Combine model judgements using ensemble strategies (e.g., majority vote)."""
    
    # Resolve tags if any input starts with @ (already validated by RunInputParamType)
    from llm_ensemble.libs.runtime.tag_manager import TagManager
    resolved_run_names = [TagManager.resolve_input(rn, "infer") for rn in input_run_names]
    
    # Load configurations
    ensemble_config = load_ensemble_config(ensemble_cfg)
    io_config = load_io_config(io_cfg, cli_name="aggregate")
    logging_config = load_logging_config(log_cfg or "observability")
    
    # Parse and route overrides if provided
    if override:
        overrides = parse_and_route_overrides(override)
        
        # Apply routed overrides to each config
        if overrides.get('ensemble'):
            ensemble_config = apply_overrides(ensemble_config, overrides['ensemble'])
        if overrides.get('io'):
            io_config = apply_overrides(io_config, overrides['io'])
    
    # Run aggregation with final configs
    run_aggregation(
        ensemble_config=ensemble_config,
        io_config=io_config,
        logging_config=logging_config,
        input_run_names=resolved_run_names,
        ensemble_config_name=ensemble_cfg,
        io_config_name=io_cfg,
        run_name=run_name,
        official=official,
        notes=notes,
        tag=tag,
    )


if __name__ == "__main__":
    app()
