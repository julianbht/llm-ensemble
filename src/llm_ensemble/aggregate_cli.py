"""Aggregate CLI - Combine judgements using ensemble strategies."""

from __future__ import annotations
from pathlib import Path
import typer

from llm_ensemble.aggregate.orchestrator import run_aggregation
from llm_ensemble.aggregate.config_loaders import load_ensemble_config
from llm_ensemble.libs.config import load_io_config
from llm_ensemble.libs.config.logging_config_loader import load_logging_config
from llm_ensemble.libs.runtime.env import load_runtime_config
from llm_ensemble.libs.runtime.path_manager import PathManager
from llm_ensemble.libs.utils.config_overrides import parse_and_route_overrides, apply_overrides
from llm_ensemble.libs.cli.common_params import IoCfg, RunId, LogCfg, Official, Notes, Override

# Load runtime configuration early
load_runtime_config()

app = typer.Typer(add_completion=False, help="LLM Ensemble – aggregate CLI")


@app.command("aggregate")
def aggregate(
    # Required parameters
    io_cfg: IoCfg,
    ensemble_cfg: str = typer.Option(
        ...,
        "--ensemble-cfg",
        help=f"Ensemble config name. Configs in {PathManager.get_configs_dir/ 'ensembles'}"
    ),
    input_paths: list[Path] = typer.Argument(
        ...,
        help="Input files containing LLMJudgement records (from infer runs)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    # Optional parameters
    run_id: RunId = None,
    log_cfg: LogCfg = None,
    official: Official = False,
    notes: Notes = None,
    override: Override = [],
):
    """Aggregate multiple model judgements using ensemble strategies.
    
    Reads LLMJudgement records from multiple infer runs, groups by (query_id, docid),
    and applies aggregation strategy (e.g., majority vote) to produce consensus decisions.
    
    All behavior is explicitly configured via config files - no implicit defaults.
    
    Examples:
        # Basic usage - aggregate two infer runs with majority vote
        aggregate --ensemble-cfg majority_vote --io-cfg ndjson \\
                  artifacts/runs/infer/run1/judgements.json \\
                  artifacts/runs/infer/run2/judgements.json
        
        # With custom run ID
        aggregate --ensemble-cfg majority_vote --io-cfg ndjson \\
                  --run-id my-ensemble-run \\
                  run1/judgements.json run2/judgements.json
        
        # Override I/O adapters
        aggregate --ensemble-cfg majority_vote --io-cfg ndjson \\
                  --override io.reader=custom_reader \\
                  run1/judgements.json run2/judgements.json
    
    Override format (prefix-based routing):
        Ensemble params: --override ensemble.strategy=weighted_vote
        I/O adapters:    --override io.reader=custom_reader
        
        Prefix must be one of: ensemble, io
    """
    # Load configurations
    ensemble_config = load_ensemble_config(ensemble_cfg)
    io_config = load_io_config(io_cfg, cli_name="aggregate")
    logging_config = load_logging_config(log_cfg or "default")
    
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
        input_files=input_paths,
        ensemble_config_name=ensemble_cfg,
        io_config_name=io_cfg,
        run_id=run_id,
        official=official,
        notes=notes,
    )


if __name__ == "__main__":
    app()

