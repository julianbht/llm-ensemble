"""Parameter definitions unique to the ingest CLI."""

from __future__ import annotations

from typing import Annotated, Optional

import typer

from llm_ensemble.libs.cli.params.types import IOConfigParamType
from llm_ensemble.libs.runtime.path_manager import PathManager

IngestIoCfg = Annotated[
    str,
    typer.Option(
        ...,
        "--io-cfg",
        click_type=IOConfigParamType("ingest"),
        help=f"I/O config name. Configs in {(PathManager.get_configs_dir() / 'io' / 'ingest').relative_to(PathManager.get_project_root())}",
    ),
]

Limit = Annotated[
    Optional[int],
    typer.Option(
        "--limit",
        "-n",
        help="Process at most this many samples (None = all)",
    ),
]
