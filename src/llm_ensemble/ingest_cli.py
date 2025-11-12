from __future__ import annotations
import typer

from llm_ensemble.ingest.orchestrator import run_ingest
from llm_ensemble.libs.config import load_io_config
from llm_ensemble.libs.config.logging_config_loader import load_logging_config
from llm_ensemble.libs.runtime.env import load_runtime_config
from llm_ensemble.libs.utils.config_overrides import parse_and_route_overrides, apply_overrides
from llm_ensemble.libs.cli.common_params import InputPath, IoCfg, RunName, LogCfg, Official, Notes, Limit, Override

# Load runtime configuration early
load_runtime_config()

app = typer.Typer(
    add_completion=True,
    help="LLM Ensemble – data ingest CLI",
    pretty_exceptions_enable=False,  # Disable Rich verbose tracebacks
)


@app.command("ingest")
def ingest(
    input_path: InputPath,
    io_cfg: IoCfg,
    limit: Limit = None,
    run_name: RunName = None,
    log_cfg: LogCfg = None,
    official: Official = False,
    notes: Notes = None,
    override: Override = [],
):
    """Normalize raw IR datasets into JudgingSample records."""
    
    # Load configurations
    io_config = load_io_config(io_cfg, cli_name="ingest")
    logging_config = load_logging_config(log_cfg or "default")

    # Parse and route overrides if provided
    if override:
        overrides = parse_and_route_overrides(override, valid_prefixes=['io'])

        # Apply overrides to I/O config
        if overrides['io']:
            io_config = apply_overrides(io_config, overrides['io'])

    # Run ingest with final config
    run_ingest(
        io_config=io_config,
        logging_config=logging_config,
        io_config_name=io_cfg,
        input_path=input_path,
        run_name=run_name,
        limit=limit,
        official=official,
        notes=notes,
    )


if __name__ == "__main__":
    app()
