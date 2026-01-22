"""Ingest CLI - Driving Adapter

CLI Layer - Driving Adapter

This is a thin driving adapter that:
1. Parses CLI arguments
2. Calls the dependency configurator to build the application
3. Executes the application via its driving port (ForRunningIngest)
"""

from __future__ import annotations
import typer

from llm_ensemble.ingest.startup.dependency_configurator import build_application
from llm_ensemble.libs.runtime.run_name import generate_run_name
from llm_ensemble.libs.cli.params.shared_params import (
    InputPath,
    RunName,
    Official,
    Notes,
)

from llm_ensemble.libs.cli.params.ingest import (
    Limit,
    IngestIoCfg,
)

app = typer.Typer(
    add_completion=True,
    help="LLM Ensemble – data ingest CLI",
    pretty_exceptions_enable=False,  # Disable verbose tracebacks
)


@app.command("ingest")
def ingest(
    # Required parameters
    input_path: InputPath,
    io_cfg: IngestIoCfg,
    # Optional parameters
    limit: Limit = None,
    run_name: RunName = None,
    official: Official = False,
    notes: Notes = None,
):
    """
    Thin CLI driving adapter that builds the application and executes it.
    """
    # Generate run name if not given
    if run_name is None:
        run_name = generate_run_name()

    # Build application
    application = build_application(
        io_name=io_cfg,
        run_name=run_name,
        official=official,
    )

    # Run application
    application.run_ingest(
        input_path=input_path,
        limit=limit,
        official=official,
        notes=notes,
    )


if __name__ == "__main__":
    app()
