"""Centralized management of all default project paths.

This module provides a single source of truth for all default directory
locations in the LLM Ensemble project. All path-related logic should use
these functions rather than computing paths independently.
"""

from __future__ import annotations
from datetime import datetime
from pathlib import Path


class PathManager:
    """Centralized path management for the LLM Ensemble project.

    All methods are static since paths are determined by project structure,
    not runtime state. This provides a single source of truth for all
    default locations.

    Example:
        >>> PathManager.get_project_root()
        PosixPath('/home/user/llm-ensemble')
        >>> PathManager.get_io_configs_dir()
        PosixPath('/home/user/llm-ensemble/configs/io')
    """

    @staticmethod
    def get_project_root() -> Path:
        """Get the project root directory.

        Computed from this file's location in src/llm_ensemble/libs/runtime/.
        Project root is 4 levels up from this file.

        Returns:
            Path to project root
        """
        return Path(__file__).parents[4]

    @staticmethod
    def get_configs_dir() -> Path:
        """Get the configs/ directory.

        Returns:
            Path to configs directory
        """
        return PathManager.get_project_root() / "configs"

    @staticmethod
    def get_io_configs_dir(cli_name: str) -> Path:
        """Get the configs/io/{cli_name}/ directory.

        Args:
            cli_name: CLI name (e.g., "ingest", "infer", "aggregate", "evaluate")

        Returns:
            Path to CLI-specific I/O configs directory

        Example:
            >>> PathManager.get_io_configs_dir("ingest")
            PosixPath('/home/user/llm-ensemble/configs/io/ingest')
        """
        return PathManager.get_configs_dir() / "io" / cli_name

    @staticmethod
    def get_model_configs_dir() -> Path:
        """Get the configs/models/ directory.

        Returns:
            Path to model configs directory
        """
        return PathManager.get_configs_dir() / "models"

    @staticmethod
    def get_prompts_dir() -> Path:
        """Get the configs/prompts/ directory.

        Returns:
            Path to prompts directory
        """
        return PathManager.get_configs_dir() / "prompts"

    @staticmethod
    def get_ensembles_dir() -> Path:
        """Get the configs/ensembles/ directory.

        Returns:
            Path to ensembles configs directory
        """
        return PathManager.get_configs_dir() / "ensembles"

    @staticmethod
    def get_generated_schemas_dir() -> Path:
        """Get the libs/generated_schemas/ directory.

        Returns:
            Path to generated schemas directory (for auto-generated JSON schemas)
        """
        return PathManager.get_project_root() / "src" / "llm_ensemble" / "libs" / "generated_schemas"

    @staticmethod
    def get_prompt_templates_dir() -> Path:
        """Get the infer/adapters/prompts/templates/ directory.

        Returns:
            Path to prompt templates directory where Jinja2 templates are stored
        """
        return PathManager.get_project_root() / "src" / "llm_ensemble" / "infer" / "adapters" / "prompts" / "templates"

    @staticmethod
    def get_artifacts_dir() -> Path:
        """Get the artifacts/ directory.

        Returns:
            Path to artifacts directory
        """
        return PathManager.get_project_root() / "artifacts"

    @staticmethod
    def get_run_dir(
        cli_name: str,
        run_id: str,
        official: bool = False
    ) -> Path:
        """Get the directory path for a CLI run.

        Run directories follow the pattern:
        - Test runs: artifacts/runs/{cli_name}/test/{run_id}/
        - Official runs: artifacts/runs/{cli_name}/official/{run_id}/

        Args:
            cli_name: CLI name (e.g., "ingest", "infer", "aggregate", "evaluate")
            run_id: Run identifier (e.g., "20250128_143022_gpt-oss-20b")
            official: If True, place in official/ subdirectory for git-tracked runs

        Returns:
            Path to run directory

        Example:
            >>> PathManager.get_run_dir("infer", "20250128_143022_gpt", official=False)
            PosixPath('artifacts/runs/infer/test/20250128_143022_gpt')
            >>> PathManager.get_run_dir("ingest", "20250128_baseline", official=True)
            PosixPath('artifacts/runs/ingest/official/20250128_baseline')
        """
        run_type = "official" if official else "test"
        return PathManager.get_artifacts_dir() / "runs" / cli_name / run_type / run_id
