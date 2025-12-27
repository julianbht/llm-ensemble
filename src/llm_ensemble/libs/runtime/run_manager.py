"""Run manager for CLI executions.

Encapsulates run name generation and directory creation concerns,
separating infrastructure setup from application logic.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
from pydantic import BaseModel

from llm_ensemble.libs.runtime.path_manager import PathManager
from llm_ensemble.libs.runtime.run_name import generate_run_name


class RunManager(ABC):
    """Abstract interface for run directory management.

    Responsibilities:
    - Generate or accept run names
    - Create and provide access to run directories
    - Enable dependency injection for test isolation
    """

    @property
    @abstractmethod
    def run_name(self) -> str:
        """Run identifier for this execution."""
        pass

    @property
    @abstractmethod
    def run_dir(self) -> Path:
        """Run directory path (created lazily on first access)."""
        pass


class ProductionRunManager(RunManager):
    """Production run manager for creating proper run directories."""

    def __init__(
        self,
        cli_name: str,
        name_hints: Optional[list[str]] = None,
        custom_run_name: Optional[str] = None,
        official: bool = False,
    ):
        """Initialize production run manager.

        Args:
            cli_name: CLI name (e.g., "ingest", "infer", "aggregate", "evaluate")
            name_hints: Optional hints for run name generation (e.g., model name, prompt)
            custom_run_name: Optional custom run name (overrides generation from hints)
            official: If True, create in official/ subdirectory for git-tracked runs
        """
        self._cli_name = cli_name
        self._official = official

        if custom_run_name:
            self._run_name = custom_run_name
        else:
            self._run_name = generate_run_name(name_hints)

        self._run_dir: Optional[Path] = None

    @property
    def run_name(self) -> str:
        return self._run_name

    @property
    def run_dir(self) -> Path:
        """Get run directory, creating it lazily on first access."""
        if self._run_dir is None:
            self._run_dir = PathManager.get_run_dir(
                self._cli_name,
                self._run_name,
                self._official
            )
            self._run_dir.mkdir(parents=True, exist_ok=True)
        return self._run_dir


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
