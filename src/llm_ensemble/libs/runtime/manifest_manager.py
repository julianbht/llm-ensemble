"""Manifest management using the Builder pattern.

Provides a ManifestBuilder for constructing CLI-specific manifests step-by-step.
The builder separates manifest construction from the final Pydantic representation.
Domain services can mutate the built manifest before it is finalized.

Note: This module no longer handles run ID generation or directory creation.
Use libs/runtime/run_id.py and libs/runtime/path_manager.py instead.
"""

from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from llm_ensemble.libs.runtime.git_utils import get_git_info


class ManifestBuilder:
    """Builder for constructing CLI-specific manifests step-by-step.

    This implements the Builder pattern, allowing orchestrators to:
    1. Initialize with base metadata (run_id, run_dir, git info)
    2. Add CLI-specific fields incrementally
    3. Finalize to create the immutable Pydantic manifest

    The builder is purely focused on manifest construction - it does not
    create run IDs or directories. Those should be created by the orchestrator
    using PathManager and generate_run_id().

    Example:
        >>> from llm_ensemble.libs.runtime.run_id import generate_run_id
        >>> from llm_ensemble.libs.runtime.path_manager import PathManager
        >>>
        >>> run_id = generate_run_id("llm-judge")
        >>> run_dir = PathManager.get_run_dir("ingest", run_id, official=False)
        >>> run_dir.mkdir(parents=True, exist_ok=True)
        >>>
        >>> builder = ManifestBuilder(
        ...     run_id=run_id,
        ...     run_dir=run_dir,
        ...     cli_name="ingest",
        ...     official=False,
        ...     notes="Testing dataset loader"
        ... )
        >>> builder.add("io_config_name", "llm_judge_ingest")
        >>> builder.add("limit", 100)
        >>> manifest = builder.finalize(IngestManifest)
    """

    def __init__(
        self,
        run_id: str,
        run_dir: Path,
        cli_name: str,
        official: bool = False,
        notes: str | None = None,
    ):
        """Initialize manifest builder with base metadata.

        Args:
            run_id: Run identifier (e.g., "20250128_143022_gpt-oss-20b")
            run_dir: Run directory path (should already exist)
            cli_name: CLI name (e.g., "ingest", "infer", "aggregate", "evaluate")
            official: If True, mark as official run
            notes: Optional notes about this run (experiment purpose, hypothesis, etc.)
        """
        self.run_id = run_id
        self.run_dir = run_dir

        # Capture git info for reproducibility
        git_info = get_git_info()

        # Initialize fields with base metadata
        self._fields: dict[str, Any] = {
            "run_id": run_id,
            "run_type": "official" if official else "test",
            "cli_name": cli_name,
            "start_time": None,  # Set by domain service when processing begins
            "end_time": None,  # Set during finalize()
            "notes": notes,
            "git_sha": git_info["git_sha"],
            "git_clean": git_info["git_clean"],
            "git_branch": git_info["git_branch"],
        }

    def add(self, key: str, value: Any) -> "ManifestBuilder":
        """Add a CLI-specific field to the manifest.

        Args:
            key: Field name
            value: Field value

        Returns:
            Self for method chaining (Fluent Builder pattern)

        Example:
            >>> builder.add("io_config_name", "ndjson").add("limit", 100)
        """
        self._fields[key] = value
        return self


    def finalize(self, manifest_class: type[BaseModel]) -> BaseModel:
        """Finalize the manifest by setting end_time and creating the Pydantic object.

        Args:
            manifest_class: The Pydantic model class to instantiate (e.g., IngestManifest)

        Returns:
            Immutable Pydantic manifest object

        Example:
            >>> manifest = builder.finalize(IngestManifest)
        """
        # Set end_time to mark completion
        self._fields["end_time"] = datetime.now()

        # Create and validate Pydantic manifest
        return manifest_class(**self._fields)


def write_standalone_manifest(manifest: BaseModel, run_dir: Path) -> Path:
    """Write a standalone manifest.json for human convenience (optional).

    NOTE: This is for quick inspection only. The source of truth for manifests
    is embedded in the domain data (e.g., JudgingSamples contain manifest refs).
    I/O adapters are responsible for persisting manifests with domain data.

    This function is provided as a convenience for:
    - Quick inspection of run metadata without loading all samples
    - Debugging and exploration
    - Compatibility with tools expecting manifest.json files

    Args:
        manifest: Pydantic Manifest object (base or CLI-specific subclass)
        run_dir: Run directory path

    Returns:
        Path to the written manifest file

    Example:
        >>> # Optionally write standalone manifest for convenience
        >>> manifest_path = write_standalone_manifest(manifest, run_dir)
    """
    # Ensure run directory exists
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write manifest as JSON
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    return manifest_path


