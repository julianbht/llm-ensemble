"""Manifest management using the Builder pattern.

Provides a ManifestBuilder for constructing CLI-specific manifests step-by-step.
The builder separates manifest construction from the final Pydantic representation.
"""

from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

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


class ManifestBuilder:
    """Builder for constructing CLI-specific manifests step-by-step.

    This implements the Builder pattern, allowing orchestrators to:
    1. Initialize with base metadata (run_id, git info, timestamps)
    2. Add CLI-specific fields incrementally
    3. Finalize to create the immutable Pydantic manifest

    The builder keeps orchestration logic in the orchestrator while providing
    a clean API for manifest construction.

    Example:
        >>> builder = ManifestBuilder(
        ...     cli_name="ingest",
        ...     name_hint="llm-judge",
        ...     official=False,
        ...     notes="Testing dataset loader"
        ... )
        >>> builder.add("io_config_name", "llm_judge_ingest")
        >>> builder.add("limit", 100)
        >>> builder.add("sample_count", 1523)
        >>> manifest = builder.finalize(IngestManifest)
    """

    def __init__(
        self,
        cli_name: str,
        name_hint: str,
        run_id: Optional[str] = None,
        official: bool = False,
        notes: Optional[str] = None,
        base_dir: Optional[Path] = None,
    ):
        """Initialize manifest builder with base metadata.

        Args:
            cli_name: CLI name (e.g., "ingest", "infer", "aggregate", "evaluate")
            name_hint: Hint for run ID generation (e.g., dataset name, model name)
            run_id: Optional custom run ID (auto-generates if not provided)
            official: If True, mark as official run (saved to official/ subdirectory)
            notes: Optional notes about this run (experiment purpose, hypothesis, etc.)
            base_dir: Base artifacts directory (defaults to ./artifacts)
        """
        # Generate or use provided run_id
        self.run_id = run_id or create_run_id(name_hint)

        # Create run directory
        self.run_dir = get_run_dir(self.run_id, cli_name, official, base_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Capture git info for reproducibility
        git_info = get_git_info()

        # Initialize fields with base metadata
        self._fields: dict[str, Any] = {
            "run_id": self.run_id,
            "run_type": "official" if official else "test",
            "cli_name": cli_name,
            "start_time": datetime.now(),
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


def initialize_manifest(
    cli_name: str,
    name_hint: str,
    run_id: Optional[str] = None,
    official: bool = False,
    notes: Optional[str] = None,
    base_dir: Optional[Path] = None,
) -> ManifestBuilder:
    """Factory function to initialize a ManifestBuilder.

    This is the primary entry point for orchestrators to start building manifests.

    Args:
        cli_name: CLI name (e.g., "ingest", "infer", "aggregate", "evaluate")
        name_hint: Hint for run ID generation (e.g., dataset name, model name)
        run_id: Optional custom run ID (auto-generates if not provided)
        official: If True, mark as official run (saved to official/ subdirectory)
        notes: Optional notes about this run (experiment purpose, hypothesis, etc.)
        base_dir: Base artifacts directory (defaults to ./artifacts)

    Returns:
        ManifestBuilder instance ready for adding CLI-specific fields

    Example:
        >>> builder = initialize_manifest(
        ...     cli_name="ingest",
        ...     name_hint="llm-judge",
        ...     official=False,
        ... )
        >>> builder.add("io_config_name", "llm_judge_ingest")
        >>> manifest = builder.finalize(IngestManifest)
    """
    return ManifestBuilder(
        cli_name=cli_name,
        name_hint=name_hint,
        run_id=run_id,
        official=official,
        notes=notes,
        base_dir=base_dir,
    )


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
    import json

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
