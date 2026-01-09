"""Common parameter definitions shared across CLIs."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from llm_ensemble.libs.runtime.path_manager import PathManager
from llm_ensemble.libs.cli.params.types import (
    LogConfigParamType,
    ModelConfigParamType,
    PromptTemplateParamType,
    RetryConfigParamType,
)

InputPath = Annotated[
    Optional[Path],
    typer.Option(
        "--input",
        "-i",
        help="Input path (optional for database-backed readers)",
    ),
]

RunName = Annotated[
    Optional[str],
    typer.Option(
        "--run-name",
        help="Custom run name (auto-generates if not provided)",
    ),
]

LogCfg = Annotated[
    str,
    typer.Option(
        "--log-cfg",
        click_type=LogConfigParamType(),
        help=(
            "Logging config name. Configs located in "
            f"{(PathManager.get_configs_dir() / 'logging').relative_to(PathManager.get_project_root())}."
        ),
    ),
]

Official = Annotated[
    bool,
    typer.Option(
        "--official",
        help="Mark as official run (saved to official/ subdirectory for git tracking)",
    ),
]

Notes = Annotated[
    Optional[str],
    typer.Option(
        "--notes",
        help="Notes about this run (experiment purpose, hypothesis, etc.)",
    ),
]

StartIdx = Annotated[
    Optional[int],
    typer.Option(
        "--start-idx",
        help="Start index into NormalizedDataset (0-indexed, inclusive, None = start from beginning)",
    ),
]

EndIdx = Annotated[
    Optional[int],
    typer.Option(
        "--end-idx",
        help="End index into NormalizedDataset (exclusive, None = process until end)",
    ),
]

RetryCfg = Annotated[
    str,
    typer.Option(
        "--retry-cfg",
        click_type=RetryConfigParamType(),
        help=f"Retry config name. Configs in {PathManager.get_retries_dir().relative_to(PathManager.get_project_root())}",
    ),
]

ModelCfg = Annotated[
    str,
    typer.Option(
        ...,
        "--model-cfg",
        click_type=ModelConfigParamType(),
        help=f"Model config name. Configs in {PathManager.get_model_configs_dir().relative_to(PathManager.get_project_root())}",
    ),
]

PromptTemplate = Annotated[
    str,
    typer.Option(
        ...,
        "--prompt-template",
        click_type=PromptTemplateParamType(),
        help="Prompt template name (bundles builder and parser, e.g., 'thomas-simple')",
    ),
]
