"""Parameter definitions unique to the infer CLI."""

from __future__ import annotations

from typing import Annotated

import typer

from llm_ensemble.libs.cli.params.types import IOConfigParamType, RunInputParamType
from llm_ensemble.libs.runtime.path_manager import PathManager

InferIoCfg = Annotated[
    str,
    typer.Option(
        ...,
        "--io-cfg",
        click_type=IOConfigParamType("infer"),
        help=f"I/O config name. Configs in {(PathManager.get_configs_dir() / 'io' / 'infer').relative_to(PathManager.get_project_root())}",
    ),
]

IngestRunInput = Annotated[
    str,
    typer.Option(
        ...,
        "--input",
        "-i",
        click_type=RunInputParamType("ingest"),
        help="Ingest run to read samples from. Use run name or @tag (e.g., '@my-experiment')",
    ),
]
