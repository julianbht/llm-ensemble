from __future__ import annotations
import typer

from llm_ensemble.ingest.application.orchestrator import run_ingest
from llm_ensemble.libs.config.io_config_loader import IOConfigFactory
from llm_ensemble.libs.runtime.env import load_runtime_config
from llm_ensemble.libs.cli.params import (
    InputPath,
    RunName,
    Official,
    Notes,
    Limit,
    IngestIoCfg,
    Tag,
)

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
    io_cfg: IngestIoCfg,
    limit: Limit = None,
    run_name: RunName = None,
    official: Official = False,
    notes: Notes = None,
    tag: Tag = None,
):
    """Normalize raw IR datasets into JudgingSample records."""

    # Load I/O configuration
    io_config = IOConfigFactory.load(io_cfg, cli_name="ingest")

    # Run ingest
    run_ingest(
        io_config=io_config,
        io_config_name=io_cfg,
        input_path=input_path,
        run_name=run_name,
        limit=limit,
        official=official,
        notes=notes,
        tag=tag,
    )


if __name__ == "__main__":
    app()
