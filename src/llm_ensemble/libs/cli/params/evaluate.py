"""Evaluate-specific CLI parameter definitions."""

from __future__ import annotations
from typing import Annotated
import typer

EvaluateIoCfg = Annotated[
    str,
    typer.Option(
        ...,
        "--io-cfg",
        help="I/O configuration name for evaluate pipeline (e.g., 'db_to_html')",
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
