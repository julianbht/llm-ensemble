"""Evaluate CLI - Driving Adapter

CLI Layer - Driving Adapter

This is a thin driving adapter that:
1. Parses CLI arguments
2. Calls the dependency configurator to build the application
3. Executes the application via its driving port (ForRunningEvaluation)
"""

from __future__ import annotations
import typer

from llm_ensemble.evaluate.startup.dependency_configurator import build_application
from llm_ensemble.libs.runtime.run_name import generate_run_name
from llm_ensemble.libs.cli.params.shared_params import (
    RunName,
    Official,
    Notes,
)

from llm_ensemble.libs.cli.params.evaluate import (
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
    """
    Thin CLI driving adapter that builds the application and executes it.
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
