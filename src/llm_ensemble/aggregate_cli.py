"""Aggregate CLI - Combine judgements using ensemble strategies."""

from __future__ import annotations
import typer

from llm_ensemble.aggregate.startup.composition_root import build_application
from llm_ensemble.libs.runtime.run_name import generate_run_name
from llm_ensemble.libs.runtime.tag_manager import TagManager
from llm_ensemble.libs.cli.params import (
    RunName,
    Official,
    Notes,
    AggregationStrategy,
    AggregateIoCfg,
    InferRunInput,
    Tag,
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
    tag: Tag = None,
):
    """Combine model judgements using ensemble strategies (e.g., majority vote)."""

    # Resolve tags if any input starts with @ (already validated by RunInputParamType)
    resolved_run_names = [TagManager.resolve_input(rn, "infer") for rn in input_run_names]

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
        tag=tag,
    )

    # Run application
    application.run_aggregation(
        run_names=resolved_run_names,
        io_config_name=io_cfg,
        run_name=run_name,
        official=official,
        notes=notes,
    )


if __name__ == "__main__":
    app()
