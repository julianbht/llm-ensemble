"""Run management for artifact organization and reproducibility.

Manages run IDs, artifact directories, and manifest files.
"""

from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from llm_ensemble.libs.runtime.git_utils import get_git_info


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


def get_run_dir(run_id: str, base_dir: Optional[Path] = None) -> Path:
    """Get the directory path for a run.

    Args:
        run_id: Run identifier
        base_dir: Base artifacts directory (defaults to ./artifacts)

    Returns:
        Path to run directory (artifacts/runs/<run_id>/)

    Example:
        >>> get_run_dir("20250115_143022_gpt-oss-20b")
        PosixPath('artifacts/runs/20250115_143022_gpt-oss-20b')
    """
    if base_dir is None:
        # Default to artifacts/ in project root (4 levels up from this file)
        base_dir = Path(__file__).parents[4] / "artifacts"

    return base_dir / "runs" / run_id


def write_manifest(
    run_dir: Path,
    cli_name: str,
    cli_args: dict[str, Any],
    metadata: dict[str, Any],
) -> Path:
    """Write a manifest.json file for a run.

    Args:
        run_dir: Run directory path
        cli_name: Name of the CLI (e.g., "ingest", "infer")
        cli_args: Command-line arguments as dict
        metadata: Additional metadata (dataset, model, counts, etc.)

    Returns:
        Path to the written manifest file

    Example:
        >>> manifest_path = write_manifest(
        ...     run_dir,
        ...     "infer",
        ...     {"model": "gpt-oss-20b", "input": "..."},
        ...     {"inference_count": 10, "avg_latency_ms": 234.5}
        ... )
    """
    # Ensure run directory exists
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build manifest
    manifest = {
        "run_id": run_dir.name,
        "cli_name": cli_name,
        "timestamp": datetime.now().isoformat(),
        "cli_args": cli_args,
        **get_git_info(),
        **metadata,
    }

    # Write to file
    manifest_path = run_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path


def load_manifest(run_id: str, base_dir: Optional[Path] = None) -> dict[str, Any]:
    """Load a manifest.json file for a run.

    Args:
        run_id: Run identifier
        base_dir: Base artifacts directory (defaults to ./artifacts)

    Returns:
        Manifest dict

    Raises:
        FileNotFoundError: If manifest doesn't exist

    Example:
        >>> manifest = load_manifest("20250115_143022_gpt-oss-20b")
        >>> manifest["cli_name"]
        'infer'
    """
    run_dir = get_run_dir(run_id, base_dir)
    manifest_path = run_dir / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_runs(base_dir: Optional[Path] = None) -> list[str]:
    """List all run IDs in the artifacts directory.

    Args:
        base_dir: Base artifacts directory (defaults to ./artifacts)

    Returns:
        List of run IDs (sorted by timestamp, newest first)

    Example:
        >>> list_runs()
        ['20250115_143055_gpt-oss-20b', '20250115_143022_llm-judge']
    """
    if base_dir is None:
        base_dir = Path(__file__).parents[4] / "artifacts"

    runs_dir = base_dir / "runs"
    if not runs_dir.exists():
        return []

    # Get all subdirectories
    run_ids = [d.name for d in runs_dir.iterdir() if d.is_dir()]

    # Sort by timestamp (newest first)
    return sorted(run_ids, reverse=True)
