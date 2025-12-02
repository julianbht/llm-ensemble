"""Common parameter definitions shared across CLIs."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from llm_ensemble.libs.runtime.path_manager import PathManager
from llm_ensemble.libs.cli.params.types import (
    AggregationStrategyAdapterParamType,
    LogConfigParamType,
    ModelConfigParamType,
    PromptConfigParamType,
    PromptParamType,
    ParserParamType,
    RetryConfigParamType,
    RunInputParamType,
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

PromptCfg = Annotated[
    str,
    typer.Option(
        ...,
        "--prompt-cfg",
        click_type=PromptConfigParamType(),
        help=f"Prompt config name. Configs in {PathManager.get_prompts_dir().relative_to(PathManager.get_project_root())}",
    ),
]

Prompt = Annotated[
    str,
    typer.Option(
        ...,
        "--prompt",
        click_type=PromptParamType(),
        help="Prompt builder name from registry (e.g., 'thomas-simple')",
    ),
]

Parser = Annotated[
    str,
    typer.Option(
        ...,
        "--parser",
        click_type=ParserParamType(),
        help="Response parser name from registry (e.g., 'thomas-simple')",
    ),
]

AggregationStrategyAdapterSpecName = Annotated[
    str,
    typer.Option(
        ...,
        "--aggregation-strategy-cfg",
        click_type=AggregationStrategyAdapterParamType(),
        help=f"Aggregation strategy adapter spec name. Configs in {PathManager.get_strategies_dir().relative_to(PathManager.get_project_root())}",
    ),
]

Tag = Annotated[
    Optional[str],
    typer.Option(
        "--tag",
        help="Tag this run for easy reference by downstream CLIs (e.g., 'my-experiment')",
    ),
]
