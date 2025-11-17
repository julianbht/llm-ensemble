"""Aggregate CLI - Combine judgements using ensemble strategies."""

from __future__ import annotations
from pathlib import Path
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
    input_paths: list[str] = typer.Argument(
        ...,
        help="Input files containing LLMJudgement records (from infer runs). Use @tag to reference tagged runs.",
    ),
    # Optional parameters
    run_name: RunName = None,
    log_cfg: LogCfg = "observability",
    official: Official = False,
    notes: Notes = None,
    override: Override = [],
    tag: Tag = None,
):
    """Combine model judgements using ensemble strategies (e.g., majority vote)."""
    # Resolve tags in input paths if they start with @
    from llm_ensemble.libs.runtime.tag_manager import TagManager
    from llm_ensemble.libs.runtime.path_manager import PathManager
    
    resolved_paths = []
    for input_path in input_paths:
        if input_path.startswith("@"):
            # Resolve tag to run directory and get output file
            tag_name = input_path[1:]
            run_name_resolved = TagManager.resolve_tag(tag_name, "infer")
            file_path = PathManager.get_infer_output_file(run_name_resolved)
            resolved_paths.append(file_path)
        else:
            resolved_paths.append(Path(input_path))
    
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
        input_files=resolved_paths,
        ensemble_config_name=ensemble_cfg,
        io_config_name=io_cfg,
        run_name=run_name,
        official=official,
        notes=notes,
        tag=tag,
    )


if __name__ == "__main__":
    app()
