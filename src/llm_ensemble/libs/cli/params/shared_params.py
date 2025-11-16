"""Common parameter definitions shared across CLIs."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from llm_ensemble.libs.runtime.path_manager import PathManager
from llm_ensemble.libs.cli.params.types import (
    EnsembleConfigParamType,
    ModelConfigParamType,
    PromptConfigParamType,
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
    Optional[str],
    typer.Option(
        "--log-cfg",
        help=(
            "Logging config name. Configs located in "
            f"{(PathManager.get_configs_dir() / 'logging').relative_to(PathManager.get_project_root())}. "
            "Defaults to 'default' (pretty printing + log saving enabled)."
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

Override = Annotated[
    list[str],
    typer.Option(
        "--override",
        "-O",
        help=(
            "Override config values (format: key=value, e.g., 'data_dir=/custom/path'). "
            "Can be specified multiple times."
        ),
    ),
]

Limit = Annotated[
    Optional[int],
    typer.Option(
        "--limit",
        help="Process at most N examples (None = no limit)",
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

PromptCfg = Annotated[
    str,
    typer.Option(
        ...,
        "--prompt-cfg",
        click_type=PromptConfigParamType(),
        help=f"Prompt config name. Configs in {PathManager.get_prompts_dir().relative_to(PathManager.get_project_root())}",
    ),
]

EnsembleCfg = Annotated[
    str,
    typer.Option(
        ...,
        "--ensemble-cfg",
        click_type=EnsembleConfigParamType(),
        help=f"Ensemble config name. Configs in {PathManager.get_ensembles_dir().relative_to(PathManager.get_project_root())}",
    ),
]
