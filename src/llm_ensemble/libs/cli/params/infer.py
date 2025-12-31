"""Parameter definitions unique to the infer CLI."""

from __future__ import annotations

from typing import Annotated

import typer

from llm_ensemble.libs.cli.params.types import InferIOConfigParamType, ProviderParamType
from llm_ensemble.libs.runtime.path_manager import PathManager

InferIoCfg = Annotated[
    str,
    typer.Option(
        ...,
        "--io-cfg",
        click_type=InferIOConfigParamType(),
        help=f"I/O config name - bundles reader and writer",
    ),
]

IngestRunInput = Annotated[
    str,
    typer.Option(
        ...,
        "--input",
        "-i",
        help="Ingest run name to read samples from",
    ),
]

Provider = Annotated[
    str,
    typer.Option(
        ...,
        "--provider",
        click_type=ProviderParamType(),
        help="Provider name (e.g., 'openrouter', 'ollama'). Available providers from registry.",
    ),
]
