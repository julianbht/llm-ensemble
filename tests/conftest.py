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

    Example:
        >>> def test_something(tmp_path, write_file):
        ...     data_file = write_file(tmp_path, "data.txt", "content")
        ...     assert data_file.exists()
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
def tmp_artifacts(tmp_path: Path, monkeypatch) -> Path:
    """Fixture providing an isolated artifacts directory for tests.

    This fixture creates a temporary artifacts directory and patches
    the runtime module to use it instead of the real artifacts/ directory.

    Automatically cleaned up after the test completes.

    Returns:
        Path to temporary artifacts directory

    Example:
        >>> def test_cli(tmp_artifacts):
        ...     # CLI will write to tmp_artifacts/runs/ingest/test/...
        ...     result = runner.invoke(app, [...])
        ...     assert (tmp_artifacts / "runs").exists()
    """
    artifacts_dir = tmp_path / "test_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Patch the runtime module to use this temporary directory
    # This affects get_run_dir() and related functions
    import llm_ensemble.libs.runtime.run_manager as run_manager

    # Store original function
    original_get_run_dir = run_manager.get_run_dir

    # Create patched version that uses tmp artifacts
    def patched_get_run_dir(run_id: str, cli_name: str, official: bool = False, base_dir: Path | None = None):
        return original_get_run_dir(run_id, cli_name, official, base_dir=artifacts_dir)

    # Apply patch
    monkeypatch.setattr(run_manager, "get_run_dir", patched_get_run_dir)

    yield artifacts_dir

    # Cleanup happens automatically via tmp_path fixture


@pytest.fixture
def mock_samples(write_file, tmp_path: Path) -> Path:
    """Fixture providing mock JudgingExample samples for inference tests.

    Creates a samples.ndjson file with 2 sample records.

    Returns:
        Path to samples.ndjson file

    Example:
        >>> def test_infer(mock_samples):
        ...     result = runner.invoke(app, ["--input", str(mock_samples)])
    """
    samples = [
        {
            "dataset": "llm-judge-2024",
            "query_id": "q1",
            "query_text": "What is AI?",
            "docid": "d1",
            "doc": "AI is artificial intelligence...",
            "gold_relevance": 2,
        },
        {
            "dataset": "llm-judge-2024",
            "query_id": "q2",
            "query_text": "What is ML?",
            "docid": "d2",
            "doc": "ML is machine learning...",
            "gold_relevance": 1,
        },
    ]

    samples_file = tmp_path / "samples.ndjson"
    with open(samples_file, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    return samples_file


@pytest.fixture
def mock_llm_judge_dataset(write_file, tmp_path: Path) -> Path:
    """Fixture providing a minimal LLM Judge dataset structure.

    Creates the 3 required files:
    - llm4eval_query_2024.txt
    - llm4eval_document_2024.jsonl
    - llm4eval_test_qrel_2024.txt

    Returns:
        Path to directory containing the dataset files

    Example:
        >>> def test_ingest(mock_llm_judge_dataset):
        ...     result = runner.invoke(app, ["--data-dir", str(mock_llm_judge_dataset)])
    """
    data_dir = tmp_path / "llm_judge_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Query file
    write_file(data_dir, "llm4eval_query_2024.txt", "q1\tWhat is AI?\n")

    # Document file
    write_file(
        data_dir,
        "llm4eval_document_2024.jsonl",
        json.dumps({"docid": "d1", "doc": "AI is artificial intelligence..."}) + "\n",
    )

    # Qrel file
    write_file(data_dir, "llm4eval_test_qrel_2024.txt", "q1 1 d1\n")

    return data_dir


@pytest.fixture
def mock_judgements(tmp_path: Path) -> Path:
    """Fixture providing mock Judgement records for testing schema validation.

    Creates a judgements.ndjson file with 2 valid judgement records.

    Returns:
        Path to judgements.ndjson file

    Example:
        >>> def test_schema(mock_judgements):
        ...     valid, invalid, errors = validate_ndjson_file(mock_judgements, "judgement")
        ...     assert valid == 2
    """
    judgements = [
        {
            "model_id": "test-model",
            "provider": "openrouter",
            "version": None,
            "query_id": "q1",
            "docid": "d1",
            "label": 2,
            "score": 2.0,
            "confidence": 0.95,
            "rationale": "This document is highly relevant",
            "raw_text": "Label: 2\nConfidence: 0.95\nRationale: This document is highly relevant",
            "latency_ms": 234.5,
            "retries": 0,
            "cost_estimate": 0.001,
            "warnings": [],
        },
        {
            "model_id": "test-model",
            "provider": "openrouter",
            "version": None,
            "query_id": "q2",
            "docid": "d2",
            "label": 1,
            "score": 1.0,
            "confidence": 0.78,
            "rationale": "Partially relevant",
            "raw_text": "Label: 1\nConfidence: 0.78\nRationale: Partially relevant",
            "latency_ms": 187.3,
            "retries": 0,
            "cost_estimate": 0.001,
            "warnings": [],
        },
    ]

    judgements_file = tmp_path / "judgements.ndjson"
    judgements_file.write_text(
        "\n".join(json.dumps(j) for j in judgements) + "\n",
        encoding="utf-8"
    )

    return judgements_file


@pytest.fixture(scope="session")
def thomas_prompt_template():
    """Session-scoped fixture for loading the thomas-et-al prompt template.

    Loaded once per test session for efficiency.

    Returns:
        Jinja2 Template object

    Example:
        >>> def test_prompt(thomas_prompt_template):
        ...     result = build_instruction(thomas_prompt_template, query="test", page_text="content")
    """
    from llm_ensemble.infer.config_loaders import load_prompt_template
    return load_prompt_template("thomas-et-al-prompt")


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
