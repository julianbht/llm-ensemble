"""Common CLI parameters shared across all LLM Ensemble CLIs.

This module provides reusable parameter definitions using Python's Annotated type hints,
allowing consistent parameter definitions across ingest, infer, aggregate, and evaluate CLIs.

With Annotated pattern, defaults go in function signatures, not in typer.Option().
"""

from pathlib import Path
from typing import Annotated, Optional
import typer
from llm_ensemble.libs.runtime.path_manager import PathManager
from llm_ensemble.libs.cli.param_types import (
    ModelConfigParamType,
    PromptConfigParamType,
    EnsembleConfigParamType,
)

# Common parameters shared by all CLIs
# Note: Defaults are specified in function signatures, not here
InputPath = Annotated[
    Optional[Path],
    typer.Option(
        "--input",
        "-i",
        help="Input path (optional for database-backed readers)",
    )
]

IoCfg = Annotated[
    str,
    typer.Option(
        "--io-cfg",
        help=f"I/O config name. Configs located in {(PathManager.get_configs_dir() / 'io' / '<cli_name>').relative_to(PathManager.get_project_root())}"
    )
]

RunName = Annotated[
    Optional[str],
    typer.Option(
        "--run-name",
        help="Custom run name (auto-generates if not provided)"
    )
]

LogCfg = Annotated[
    Optional[str],
    typer.Option(
        "--log-cfg",
        help=f"Logging config name. Configs located in {(PathManager.get_configs_dir() / 'logging').relative_to(PathManager.get_project_root())}. Defaults to 'default' (pretty printing + log saving enabled)."
    )
]

Official = Annotated[
    bool,
    typer.Option(
        "--official",
        help="Mark as official run (saved to official/ subdirectory for git tracking)"
    )
]

Notes = Annotated[
    Optional[str],
    typer.Option(
        "--notes",
        help="Notes about this run (experiment purpose, hypothesis, etc.)"
    )
]

Override = Annotated[
    list[str],
    typer.Option(
        "--override",
        "-O",
        help="Override config values (format: key=value, e.g., 'data_dir=/custom/path'). Can be specified multiple times."
    )
]

Limit = Annotated[
    Optional[int],
    typer.Option(
        "--limit",
        help="Process at most N examples (None = no limit)"
    )
]

# CLI-specific parameters with validation callbacks

ModelCfg = Annotated[
    str,  # Required - non-optional type
    typer.Option(
        ...,  # Required marker
        "--model-cfg",
        click_type=ModelConfigParamType(),
        help=f"Model config name. Configs in {PathManager.get_model_configs_dir().relative_to(PathManager.get_project_root())}"
    )
]

PromptCfg = Annotated[
    str,  # Required - non-optional type
    typer.Option(
        ...,  # Required marker
        "--prompt-cfg",
        click_type=PromptConfigParamType(),
        help=f"Prompt config name. Configs in {PathManager.get_prompts_dir().relative_to(PathManager.get_project_root())}"
    )
]

EnsembleCfg = Annotated[
    str,  # Required - non-optional type
    typer.Option(
        ...,  # Required marker
        "--ensemble-cfg",
        click_type=EnsembleConfigParamType(),
        help=f"Ensemble config name. Configs in {PathManager.get_ensembles_dir().relative_to(PathManager.get_project_root())}"
    )
]
