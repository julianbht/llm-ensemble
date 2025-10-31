"""Common CLI parameters shared across all LLM Ensemble CLIs.

This module provides reusable parameter definitions using Python's Annotated type hints,
allowing consistent parameter definitions across ingest, infer, aggregate, and evaluate CLIs.

With Annotated pattern, defaults go in function signatures, not in typer.Option().
"""

from pathlib import Path
from typing import Annotated, Optional
import typer
from llm_ensemble.libs.runtime.path_manager import PathManager

# Common parameters shared by all CLIs
# Note: Defaults are specified in function signatures, not here
InputPath = Annotated[
    Path,
    typer.Option(
        "--input",
        "-i",
        help="Input path",
        exists=True,
    )
]

IoCfg = Annotated[
    str,
    typer.Option(
        "--io-cfg",
        help=f"I/O config name. Configs located in {(PathManager.get_configs_dir() / 'io' / '<cli_name>').relative_to(PathManager.get_project_root())}"
    )
]

RunId = Annotated[
    Optional[str],
    typer.Option(
        "--run-id",
        help="Custom run ID (auto-generates if not provided)"
    )
]

SaveLogs = Annotated[
    bool,
    typer.Option(
        "--save-logs",
        help="Save logs to run.log file in run directory"
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
