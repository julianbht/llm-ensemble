"""
Utility to copy generated figures to Overleaf LaTeX repository.

Import and call at the end of plotting scripts to automatically sync figures.
"""

import shutil
import subprocess
from pathlib import Path


# Hardcoded Overleaf repository path (relative to project root)
OVERLEAF_FIGURES_DIR = Path("../bachelor-thesis-overleaf/figures")


def copy_figure_to_overleaf(figure_path: Path, project_root: Path = None, git_sync: bool = True) -> None:
    """Copy a generated figure to the Overleaf LaTeX repository and optionally sync with git.

    Args:
        figure_path: Path to the generated figure file
        project_root: Optional project root path. If None, infers from this file's location.
        git_sync: If True, pull before copying and push after (default: True)

    Raises:
        FileNotFoundError: If figure file doesn't exist
        RuntimeError: If Overleaf directory doesn't exist or git operations fail
    """
    if not figure_path.exists():
        raise FileNotFoundError(f"Figure file not found: {figure_path}")

    # Resolve project root if not provided
    if project_root is None:
        # This file is in scripts/figures/, so go up 2 levels
        project_root = Path(__file__).parent.parent.parent

    # Resolve Overleaf directory
    overleaf_dir = (project_root / OVERLEAF_FIGURES_DIR).resolve()
    overleaf_repo_root = overleaf_dir.parent

    if not overleaf_dir.exists():
        raise RuntimeError(
            f"Overleaf figures directory not found: {overleaf_dir}\n"
            f"Expected at: {OVERLEAF_FIGURES_DIR} relative to project root"
        )

    # Git pull before copying (if enabled)
    if git_sync:
        try:
            print("Pulling latest changes from Overleaf...")
            subprocess.run(
                ["git", "pull"],
                cwd=overleaf_repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            print("✓ Git pull successful")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Git pull failed: {e.stderr.strip()}")
            print("Continuing with copy anyway...")

    # Copy file to Overleaf
    destination = overleaf_dir / figure_path.name
    shutil.copy2(figure_path, destination)
    print(f"✓ Copied to Overleaf: {destination}")

    # Git commit and push after copying (if enabled)
    if git_sync:
        try:
            # Add the specific figure file
            subprocess.run(
                ["git", "add", f"figures/{figure_path.name}"],
                cwd=overleaf_repo_root,
                check=True,
                capture_output=True,
                text=True,
            )

            # Commit with descriptive message
            commit_message = f"Update figure: {figure_path.name}"
            subprocess.run(
                ["git", "commit", "-m", commit_message],
                cwd=overleaf_repo_root,
                check=True,
                capture_output=True,
                text=True,
            )

            # Push to remote
            print("Pushing changes to Overleaf...")
            subprocess.run(
                ["git", "push"],
                cwd=overleaf_repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            print(f"✓ Git push successful: {commit_message}")

        except subprocess.CalledProcessError as e:
            # If commit fails because there are no changes, that's okay
            if "nothing to commit" in e.stderr or "nothing to commit" in e.stdout:
                print("✓ No changes to commit (figure unchanged)")
            else:
                print(f"Warning: Git commit/push failed: {e.stderr.strip()}")
                print("Figure copied but not synced to git")
