"""Evaluate CLI - Driving Adapter

CLI Layer - Driving Adapter

This is a thin driving adapter that:
1. Parses CLI arguments
2. Calls the dependency configurator to build the application
3. Executes the application via its run_evaluation method

The application handles all backend concerns (infrastructure setup, logging,
evaluation execution, result persistence). This adapter simply triggers it
and all logging appears in the terminal automatically.

Tested via CLI integration tests.
"""

from __future__ import annotations
import typer

from llm_ensemble.evaluate.startup.composition_root import build_application
from llm_ensemble.libs.runtime.run_name import generate_run_name
from llm_ensemble.libs.cli.params import (
    RunName,
    Official,
    Notes,
    EvaluateIoCfg,
    EvaluateRunInput,
)

app = typer.Typer(
    add_completion=True,
    help="LLM Ensemble – evaluate CLI",
    pretty_exceptions_enable=False,  # Disable Rich verbose tracebacks
)


@app.command("evaluate")
def evaluate(
    # Required parameters
    io_cfg: EvaluateIoCfg,
    input_run_name: EvaluateRunInput,
    # Optional parameters
    run_name: RunName = None,
    official: Official = False,
    notes: Notes = None,
):
    """Compute evaluation metrics for model judgements.

    Thin CLI driving adapter that builds the application and executes it.
    All backend logic (infrastructure, logging, evaluation) handled by application.
    """
    # Generate run name if not given
    if run_name is None:
        name_hints = [
            io_cfg,
        ]
        run_name = generate_run_name(name_hints)

    # Build application
    application = build_application(
        io_name=io_cfg,
        run_name=run_name,
        official=official,
    )

    # Run application
    application.run_evaluation(
        input_run_name=input_run_name,
        official=official,
        notes=notes,
    )


if __name__ == "__main__":
    app()
