"""Parameter definitions unique to the aggregate CLI."""

from __future__ import annotations

from typing import Annotated

import typer

from llm_ensemble.libs.cli.params.types import IOConfigParamType
from llm_ensemble.libs.runtime.path_manager import PathManager

AggregateIoCfg = Annotated[
    str,
    typer.Option(
        ...,
        "--io-cfg",
        click_type=IOConfigParamType("aggregate"),
        help=f"I/O config name. Configs in {(PathManager.get_configs_dir() / 'io' / 'aggregate').relative_to(PathManager.get_project_root())}",
    ),
]
