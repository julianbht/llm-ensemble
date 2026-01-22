"""Aggregate CLI - Driving Adapter

CLI Layer - Driving Adapter

This is a thin driving adapter that:
1. Parses CLI arguments
2. Calls the dependency configurator to build the application
3. Executes the application via its driving port (ForRunningAggregation)
"""

from __future__ import annotations
import typer

from llm_ensemble.aggregate.startup.dependency_configurator import build_application
from llm_ensemble.libs.runtime.run_name import generate_run_name
from llm_ensemble.libs.cli.params.shared_params import (
    RunName,
    Official,
    Notes,
)
from llm_ensemble.libs.cli.params.aggregate import (
    AggregationStrategy,
    AggregateIoCfg,
    InferRunInput,
)


app = typer.Typer(
    add_completion=True,
    help="LLM Ensemble – aggregate CLI",
    pretty_exceptions_enable=False,  # Disable Rich verbose tracebacks
)


@app.command("aggregate")
def aggregate(
    # Required parameters
    aggregation_strategy: AggregationStrategy,
    io_cfg: AggregateIoCfg,
    input_run_names: InferRunInput,
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
            aggregation_strategy,
            io_cfg,
        ]
        run_name = generate_run_name(name_hints)

    # Build application
    application = build_application(
        aggregation_strategy_name=aggregation_strategy,
        io_name=io_cfg,
        run_name=run_name,
        official=official,
    )

    # Run application
    application.run_aggregation(
        input_run_names=input_run_names,
        official=official,
        notes=notes,
    )


if __name__ == "__main__":
    app()
