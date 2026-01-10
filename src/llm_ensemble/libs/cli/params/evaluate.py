"""Evaluate-specific CLI parameter definitions."""

from __future__ import annotations
from typing import Annotated
import typer

from llm_ensemble.libs.cli.params.types import EvaluateIOConfigParamType

EvaluateIoCfg = Annotated[
    str,
    typer.Option(
        ...,
        "--io-cfg",
        click_type=EvaluateIOConfigParamType(),
        help="I/O configuration name for evaluate pipeline (e.g., 'db_infer_to_json', 'db_aggregate_to_json')",
    ),
]

EvaluateRunInput = Annotated[
    str,
    typer.Option(
        ...,
        "--input",
        help="Input run name (infer or aggregate run to evaluate)",
    ),
]
