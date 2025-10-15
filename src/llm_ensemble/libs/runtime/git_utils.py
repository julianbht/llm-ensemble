"""Git utilities for reproducibility tracking.

Captures git metadata for run manifests.
"""

from __future__ import annotations
import subprocess
from pathlib import Path
from typing import Optional


def get_git_sha() -> str:
    """Get the current git commit SHA.

    Returns:
        Git commit SHA (short 7-char format), or "unknown" if not in a git repo

    Example:
        >>> get_git_sha()
        '4fd2136'
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

    Example:
        >>> get_git_status_clean()
        True
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


def get_git_info() -> dict:
    """Get comprehensive git information for manifest.

    Returns:
        Dict with git_sha, git_clean, and git_branch

    Example:
        >>> get_git_info()
        {'git_sha': '4fd2136', 'git_clean': True, 'git_branch': 'master'}
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

    return {
        "git_sha": sha,
        "git_clean": clean,
        "git_branch": branch,
    }
