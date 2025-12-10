"""Git utilities for reproducibility tracking.

Captures git metadata for run manifests.
"""

from __future__ import annotations
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class GitInfo:
    """Git metadata for reproducibility tracking."""

    git_sha: str
    git_clean: bool
    git_branch: str


def get_git_sha() -> str:
    """Get the current git commit SHA.

    Returns:
        Git commit SHA (short 7-char format), or "unknown" if not in a git repo
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"


def get_git_status_clean() -> bool:
    """Check if the git working tree is clean (no uncommitted changes).

    Returns:
        True if working tree is clean, False if there are uncommitted changes
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
        return len(result.stdout.strip()) == 0
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_git_info() -> GitInfo:
    """Get comprehensive git information for manifest.

    Returns:
        GitInfo object with SHA, clean status, and branch name
    """
    sha = get_git_sha()
    clean = get_git_status_clean()

    # Get branch name
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
        branch = result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        branch = "unknown"

    return GitInfo(
        git_sha=sha,
        git_clean=clean,
        git_branch=branch,
    )
