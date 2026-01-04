"""Aggregate CLI - Driving Adapter

CLI Layer - Driving Adapter

This is a thin driving adapter that:
1. Parses CLI arguments
2. Calls the dependency configurator to build the application
3. Executes the application via its run_aggregation method

The application handles all backend concerns (infrastructure setup, logging,
aggregation execution, result persistence). This adapter simply triggers it
and all logging appears in the terminal automatically.

Tested via CLI integration tests.
"""

from __future__ import annotations
import typer

from llm_ensemble.aggregate.startup.dependency_configurator import build_application
from llm_ensemble.libs.runtime.run_name import generate_run_name
from llm_ensemble.libs.cli.params import (
    RunName,
    Official,
    Notes,
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
    """Combine model judgements using ensemble strategies (e.g., majority vote).

    Thin CLI driving adapter that builds the application and executes it.
    All backend logic (infrastructure, logging, aggregation) handled by application.
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
