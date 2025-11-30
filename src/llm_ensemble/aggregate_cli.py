"""Aggregate CLI - Combine judgements using ensemble strategies."""

from __future__ import annotations
import typer

from llm_ensemble.aggregate.orchestrator import run_aggregation
from llm_ensemble.aggregate.config_loaders import load_aggregation_strategy_config
from llm_ensemble.libs.config import load_io_config
from llm_ensemble.libs.config.logging_config_loader import load_logging_config
from llm_ensemble.libs.runtime.env import load_runtime_config
from llm_ensemble.libs.runtime.tag_manager import TagManager
from llm_ensemble.libs.utils.config_overrides import parse_and_route_overrides, apply_overrides
from llm_ensemble.libs.cli.params import (
    RunName,
    LogCfg,
    Official,
    Notes,
    Override,
    AggregationStrategyAdapterSpecName,
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
    aggregation_strategy_adapter_spec_name: AggregationStrategyAdapterSpecName,
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
    resolved_run_names = [TagManager.resolve_input(rn, "infer") for rn in input_run_names]

    # Load configurations
    aggregation_strategy_config = load_aggregation_strategy_config(aggregation_strategy_adapter_spec_name)
    io_config = load_io_config(io_cfg, cli_name="aggregate")
    logging_config = load_logging_config(log_cfg or "observability")

    # Parse and route overrides if provided
    if override:
        overrides = parse_and_route_overrides(override)

        # Apply routed overrides to each config
        if overrides.get('aggregation_strategy'):
            aggregation_strategy_config = apply_overrides(aggregation_strategy_config, overrides['aggregation_strategy'])
        if overrides.get('io'):
            io_config = apply_overrides(io_config, overrides['io'])

    # Run aggregation with final configs
    run_aggregation(
        aggregation_strategy_config=aggregation_strategy_config,
        io_config=io_config,
        logging_config=logging_config,
        input_run_names=resolved_run_names,
        aggregation_strategy_config_name=aggregation_strategy_adapter_spec_name,
        io_config_name=io_cfg,
        run_name=run_name,
        official=official,
        notes=notes,
        tag=tag,
    )


if __name__ == "__main__":
    app()
