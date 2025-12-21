"""Simple run manager for common CLI run directory operations.

Provides utilities for creating run directories and writing summary files
that all CLIs need to do.
"""

from __future__ import annotations
from pathlib import Path
from pydantic import BaseModel

from llm_ensemble.libs.runtime.path_manager import PathManager


def create_run_directory(
    cli_name: str,
    run_name: str,
    official: bool = False,
) -> Path:
    """Create a run directory for a CLI execution.

    Args:
        cli_name: CLI name (e.g., "ingest", "infer", "aggregate", "evaluate")
        run_name: Run identifier
        official: If True, create in official/ subdirectory for git-tracked runs

    Returns:
        Path to created run directory
    """
    run_dir = PathManager.get_run_dir(cli_name, run_name, official)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_summary(summary: BaseModel, run_dir: Path) -> Path:
    """Write a summary object to summary.json in the run directory.

    Args:
        summary: Pydantic model containing run summary data
        run_dir: Run directory path

    Returns:
        Path to written summary file
    """
    summary_path = run_dir / "summary.json"
    summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    return summary_path
