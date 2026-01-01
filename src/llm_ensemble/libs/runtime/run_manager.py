"""Run manager for CLI executions.

Encapsulates run name generation and directory creation concerns,
separating infrastructure setup from application logic.
"""

from __future__ import annotations
from pathlib import Path
from pydantic import BaseModel

def persist_write_summary(summary: BaseModel, run_dir: Path) -> Path:
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
