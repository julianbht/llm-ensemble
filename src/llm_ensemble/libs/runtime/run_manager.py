"""Run management for artifact organization and reproducibility.

Manages run IDs, artifact directories, and manifest files.
"""

from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

from llm_ensemble.libs.runtime.git_utils import get_git_info
from llm_ensemble.libs.schemas.manifest import Manifest


def create_run_id(name_hint: str) -> str:
    """Generate a unique run ID.

    Format: YYYYMMDD_HHMMSS_<hint>

    Args:
        name_hint: Hint for the run (e.g., dataset name, model name)

    Returns:
        Unique run ID string

    Example:
        >>> create_run_id("gpt-oss-20b")
        '20250115_143022_gpt-oss-20b'
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sanitize name_hint (remove special chars, limit length)
    safe_hint = "".join(c for c in name_hint if c.isalnum() or c in "-_")[:30]
    return f"{timestamp}_{safe_hint}"


def create_run(
    cli_name: str,
    name_hint: str,
    official: bool = False,
    notes: Optional[str] = None,
    base_dir: Optional[Path] = None,
) -> tuple[Manifest, Path]:
    """Create a new run with base manifest and directory.

    This is the primary entry point for starting a new CLI run. It handles:
    - Generating a unique run ID
    - Creating the run directory
    - Capturing git metadata for reproducibility
    - Building the base Manifest object

    The orchestrator should extend the returned Manifest with CLI-specific fields
    before running business logic and writing the final manifest.

    Args:
        cli_name: CLI name (e.g., "ingest", "infer", "aggregate", "evaluate")
        name_hint: Hint for the run ID (e.g., dataset name, model name)
        official: If True, mark as official run (saved to official/ subdirectory)
        notes: Optional notes about this run (experiment purpose, hypothesis, etc.)
        base_dir: Base artifacts directory (defaults to ./artifacts)

    Returns:
        Tuple of (base_manifest, run_dir):
        - base_manifest: Manifest object with auto-captured metadata
        - run_dir: Path to the created run directory

    Example:
        >>> manifest, run_dir = create_run(
        ...     cli_name="ingest",
        ...     name_hint="llm-judge",
        ...     official=False,
        ...     notes="Testing new dataset loader"
        ... )
        >>> # Extend with CLI-specific fields
        >>> ingest_manifest = IngestManifest(
        ...     **manifest.model_dump(),
        ...     io_config_name="llm_judge_ingest",
        ...     limit=100,
        ... )
    """
    # Generate run ID and create directory
    run_id = create_run_id(name_hint)
    run_dir = get_run_dir(run_id, cli_name, official, base_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Capture git info for reproducibility
    git_info = get_git_info()

    # Build base manifest
    manifest = Manifest(
        run_id=run_id,
        run_type="official" if official else "test",
        cli_name=cli_name,
        start_time=datetime.now(),
        end_time=None,
        notes=notes,
        git_sha=git_info["git_sha"],
        git_clean=git_info["git_clean"],
        git_branch=git_info["git_branch"],
    )

    return manifest, run_dir


def get_run_dir(run_id: str, cli_name: str, official: bool = False, base_dir: Optional[Path] = None) -> Path:
    """Get the directory path for a run.

    Args:
        run_id: Run identifier
        cli_name: CLI name (e.g., "ingest", "infer", "aggregate", "evaluate")
        official: If True, place in official/ subdirectory (for git-tracked runs)
        base_dir: Base artifacts directory (defaults to ./artifacts)

    Returns:
        Path to run directory:
        - Test runs: artifacts/runs/<cli_name>/test/<run_id>/
        - Official runs: artifacts/runs/<cli_name>/official/<run_id>/

    Example:
        >>> get_run_dir("20250115_143022_gpt-oss-20b", "infer")
        PosixPath('artifacts/runs/infer/test/20250115_143022_gpt-oss-20b')
        >>> get_run_dir("20250115_143022_baseline", "infer", official=True)
        PosixPath('artifacts/runs/infer/official/20250115_143022_baseline')
    """
    if base_dir is None:
        # Default to artifacts/ in project root (4 levels up from this file)
        base_dir = Path(__file__).parents[4] / "artifacts"

    run_type = "official" if official else "test"
    return base_dir / "runs" / cli_name / run_type / run_id


def finalize_manifest(manifest: BaseModel) -> BaseModel:
    """Finalize a manifest by setting completion metadata.

    Updates the manifest with end_time. Call this after business logic completes
    and before persisting via I/O adapters.

    Args:
        manifest: Pydantic Manifest object (base or CLI-specific subclass)

    Returns:
        Updated manifest with end_time set

    Example:
        >>> # After business logic completes
        >>> manifest = finalize_manifest(manifest)
        >>> # Now pass to I/O adapter for persistence
        >>> service.ingest_dataset(..., manifest=manifest)
    """
    manifest.end_time = datetime.now()
    return manifest


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


def load_manifest(run_id: str, cli_name: str, base_dir: Optional[Path] = None) -> dict[str, Any]:
    """Load a manifest.json file for a run.

    Args:
        run_id: Run identifier
        cli_name: CLI name (e.g., "ingest", "infer")
        base_dir: Base artifacts directory (defaults to ./artifacts)

    Returns:
        Manifest dict

    Raises:
        FileNotFoundError: If manifest doesn't exist

    Example:
        >>> manifest = load_manifest("20250115_143022_gpt-oss-20b", "infer")
        >>> manifest["cli_name"]
        'infer'
    """
    run_dir = get_run_dir(run_id, cli_name, base_dir)
    manifest_path = run_dir / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_runs(cli_name: Optional[str] = None, base_dir: Optional[Path] = None) -> list[str]:
    """List all run IDs in the artifacts directory.

    Args:
        cli_name: Optional CLI name to filter by (e.g., "ingest", "infer")
        base_dir: Base artifacts directory (defaults to ./artifacts)

    Returns:
        List of run IDs (sorted by timestamp, newest first)

    Example:
        >>> list_runs("infer")
        ['20250115_143055_gpt-oss-20b', '20250115_143022_gpt-oss-20b']
        >>> list_runs()  # All runs from all CLIs
        ['20250115_143055_gpt-oss-20b', '20250115_143022_llm-judge', ...]
    """
    if base_dir is None:
        base_dir = Path(__file__).parents[4] / "artifacts"

    runs_dir = base_dir / "runs"
    if not runs_dir.exists():
        return []

    run_ids = []

    if cli_name:
        # List runs for specific CLI
        cli_dir = runs_dir / cli_name
        if cli_dir.exists():
            run_ids = [d.name for d in cli_dir.iterdir() if d.is_dir()]
    else:
        # List all runs from all CLIs
        for cli_dir in runs_dir.iterdir():
            if cli_dir.is_dir():
                run_ids.extend([d.name for d in cli_dir.iterdir() if d.is_dir()])

    # Sort by timestamp (newest first)
    return sorted(run_ids, reverse=True)
