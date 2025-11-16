"""Shared pytest fixtures for all test modules.

This conftest.py provides test isolation via temporary directories
and shared helpers to avoid duplication across test files.
"""

import json
from pathlib import Path
from typing import Callable

import pytest


@pytest.fixture
def write_file() -> Callable[[Path, str, str], Path]:
    """Fixture providing a helper to write test files.

    Returns:
        Function that writes content to a file and returns its path
    """
    def _write(base: Path, name: str, content: str) -> Path:
        """Write a file with given content.

        Args:
            base: Base directory
            name: Filename (can include subdirs like "subdir/file.txt")
            content: File content

        Returns:
            Path to created file
        """
        filepath = base / name
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content, encoding="utf-8")
        return filepath

    return _write


@pytest.fixture
def mock_git_info(monkeypatch):
    """Fixture that monkeypatches git_utils.get_git_info to return deterministic values.

    This ensures test reproducibility by providing consistent git metadata
    regardless of the actual git state.

    Returns:
        Dict with deterministic git metadata
    """
    deterministic_git_info = {
        "git_sha": "abc1234",
        "git_clean": True,
        "git_branch": "test-branch"
    }

    # Patch at the module level where it's defined
    monkeypatch.setattr("llm_ensemble.libs.runtime.git_utils.get_git_info", lambda: deterministic_git_info)

    return deterministic_git_info


@pytest.fixture
def tmp_runs_dir(tmp_path: Path, monkeypatch):
    """Fixture that redirects run directories to a temporary location.

    Monkeypatches PathManager.get_run_dir and get_artifacts_dir to use
    tmp_path/artifacts instead of the real artifacts directory.

    Automatically cleaned up after the test via tmp_path fixture.

    Returns:
        Tuple of (artifacts_base_path, get_run_dir_helper)
        - artifacts_base_path: Path to tmp_path/artifacts
        - get_run_dir_helper: Function(cli_name, run_name, official) -> Path
    """
    from llm_ensemble.libs.runtime.path_manager import PathManager

    # Create temporary artifacts directory
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Store original methods
    original_get_artifacts_dir = PathManager.get_artifacts_dir
    original_get_run_dir = PathManager.get_run_dir

    # Create patched versions
    def patched_get_artifacts_dir() -> Path:
        return artifacts_dir

    def patched_get_run_dir(
        cli_name: str,
        run_name: str,
        official: bool = False
    ) -> Path:
        # Call original with base_dir override would be ideal, but original doesn't accept it
        # So we reconstruct the path logic here
        run_type = "official" if official else "test"
        return artifacts_dir / "runs" / cli_name / run_type / run_name

    # Apply patches
    monkeypatch.setattr(PathManager, "get_artifacts_dir", patched_get_artifacts_dir)
    monkeypatch.setattr(PathManager, "get_run_dir", patched_get_run_dir)

    # Helper for tests to compute expected run directory
    def get_run_dir_helper(cli_name: str, run_name: str, official: bool = False) -> Path:
        return patched_get_run_dir(cli_name, run_name, official)

    yield artifacts_dir, get_run_dir_helper

    # Cleanup happens automatically via tmp_path fixture


@pytest.fixture
def fake_reader_factory():
    """Factory fixture for creating fake DatasetReader test doubles.

    Returns a factory function that creates configurable fake readers
    for testing without actual I/O.

    Returns:
        Factory function(normalized_dataset: NormalizedDataset) -> FakeReader
    """
    from llm_ensemble.ingest.ports import DatasetReader
    from llm_ensemble.ingest.schemas import NormalizedDataset

    class FakeReader(DatasetReader):
        """Test double for DatasetReader that returns pre-configured normalized dataset."""

        def __init__(self, normalized_dataset: NormalizedDataset):
            self.normalized_dataset = normalized_dataset
            self.called_with = {}

        def read(
            self,
            input_path: Path,
            limit: int | None = None,
        ) -> NormalizedDataset:
            # Record what we were called with
            self.called_with = {
                "input_path": input_path,
                "limit": limit,
            }
            # Apply limit if specified
            if limit is not None:
                limited_samples = self.normalized_dataset.samples[:limit]
                return NormalizedDataset(
                    dataset=self.normalized_dataset.dataset,
                    samples=limited_samples
                )
            return self.normalized_dataset

    return FakeReader


@pytest.fixture
def fake_writer_factory():
    """Factory fixture for creating fake DatasetWriter test doubles.

    Returns a factory function that creates configurable fake writers
    for testing without actual I/O.

    Returns:
        Factory function() -> FakeWriter
    """
    from typing import Any
    from llm_ensemble.ingest.ports import DatasetWriter
    from llm_ensemble.ingest.schemas import JudgingSample, WriteSummary, NormalizedDataset

    class FakeWriter(DatasetWriter):
        """Test double for DatasetWriter that records writes without I/O."""

        def __init__(self):
            self.written_samples: list[JudgingSample] = []
            self.write_calls: list[tuple[NormalizedDataset, Any]] = []

        def write(self, normalized_dataset: NormalizedDataset, run_info: Any) -> WriteSummary:
            # Record the call
            self.write_calls.append((normalized_dataset, run_info))
            # Store samples
            self.written_samples.extend(normalized_dataset.samples)
            # Return summary
            return WriteSummary(samples_created=normalized_dataset.sample_count)

    return FakeWriter


# Pytest configuration hooks
def pytest_configure(config):
    """Register custom markers for test categorization."""
    config.addinivalue_line(
        "markers", "unit: Unit tests (fast, isolated, no I/O)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests (may use files, adapters)"
    )
    config.addinivalue_line(
        "markers", "slow: Slow tests (API calls, long running)"
    )
    config.addinivalue_line(
        "markers", "requires_api: Tests requiring API credentials"
    )
