"""Parameter definitions unique to the aggregate CLI."""

from __future__ import annotations

from typing import Annotated

import typer

from llm_ensemble.libs.cli.params.types import IOConfigParamType, RunInputParamType
from llm_ensemble.libs.cli.params.aggregation_strategy import AggregationStrategyParamType
from llm_ensemble.libs.runtime.path_manager import PathManager

AggregationStrategy = Annotated[
    str,
    typer.Option(
        ...,
        "--aggregation-strategy",
        click_type=AggregationStrategyParamType(),
        help="Aggregation strategy (e.g., 'majority_vote')",
    ),
]

AggregateIoCfg = Annotated[
    str,
    typer.Option(
        ...,
        "--io-cfg",
        click_type=IOConfigParamType("aggregate"),
        help=f"I/O config name. Configs in {(PathManager.get_configs_dir() / 'io' / 'aggregate').relative_to(PathManager.get_project_root())}",
    ),
]

InferRunInput = Annotated[
    list[str],
    typer.Option(
        ...,
        "--input",
        "-i",
        click_type=RunInputParamType("infer"),
        help="Infer runs to read judgements from. Use run names or @tags (e.g., '@my-experiment'). Can specify multiple.",
    ),
]
