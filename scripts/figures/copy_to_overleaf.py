"""
Utility to copy generated figures to Overleaf LaTeX repository.

Import and call at the end of plotting scripts to automatically sync figures.
"""

import shutil
from pathlib import Path


# Hardcoded Overleaf repository path (relative to project root)
OVERLEAF_FIGURES_DIR = Path("../bachelor-thesis-overleaf/figures")


def copy_figure_to_overleaf(figure_path: Path, project_root: Path = None) -> None:
    """Copy a generated figure to the Overleaf LaTeX repository.

    Args:
        figure_path: Path to the generated figure file
        project_root: Optional project root path. If None, infers from this file's location.

    Raises:
        FileNotFoundError: If figure file doesn't exist
        RuntimeError: If Overleaf directory doesn't exist
    """
    if not figure_path.exists():
        raise FileNotFoundError(f"Figure file not found: {figure_path}")

    # Resolve project root if not provided
    if project_root is None:
        # This file is in scripts/figures/, so go up 2 levels
        project_root = Path(__file__).parent.parent.parent

    # Resolve Overleaf directory
    overleaf_dir = (project_root / OVERLEAF_FIGURES_DIR).resolve()

    if not overleaf_dir.exists():
        raise RuntimeError(
            f"Overleaf figures directory not found: {overleaf_dir}\n"
            f"Expected at: {OVERLEAF_FIGURES_DIR} relative to project root"
        )

    # Copy file to Overleaf
    destination = overleaf_dir / figure_path.name
    shutil.copy2(figure_path, destination)

    print(f"✓ Copied to Overleaf: {destination}")
