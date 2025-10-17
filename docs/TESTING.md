# Testing Guide

This document describes the testing framework, best practices, and how to write new tests for the LLM Ensemble project.

## Overview

The project uses **pytest** for testing with a focus on:
- **Test isolation** — Each test runs in its own temporary directory
- **No side effects** — Tests never write to real `artifacts/` directory
- **Automatic cleanup** — All test artifacts are automatically cleaned up
- **Shared fixtures** — Common test utilities are centralized in `conftest.py`
- **Clear categorization** — Tests are marked as unit, integration, slow, or requires_api

## Test Structure

The project follows the standard **src-layout** with tests at the project root:

```
llm-ensemble/
├── src/
│   └── llm_ensemble/       # Source code (no tests here)
│       ├── ingest/
│       ├── infer/
│       └── libs/
└── tests/                  # All tests at project root
    ├── conftest.py         # Shared fixtures for all tests
    ├── ingest/
    │   ├── test_llm_judge_ingest.py   # Unit tests for adapters
    │   └── test_ingest_cli.py         # Integration tests for CLI
    └── infer/
        ├── test_prompt_builder.py     # Unit tests for domain logic
        ├── test_prompt_loader.py      # Unit tests for adapters
        └── test_infer_cli.py          # Integration tests for CLI
```

## Running Tests

```bash
# Run all tests
make test

# Run specific module
make test-ingest
make test-infer

# Run by marker
pytest -m unit          # Fast unit tests only
pytest -m integration   # Integration tests only
pytest -m "not slow"    # Exclude slow tests

# Run with coverage
pytest --cov=llm_ensemble --cov-report=html

# Verbose output with print statements
pytest -v -s
```

## Shared Fixtures

All shared fixtures are defined in `tests/conftest.py`:

### `write_file`

Helper to create test files easily:

```python
def test_something(tmp_path, write_file):
    data_file = write_file(tmp_path, "data.txt", "content")
    assert data_file.exists()

    # Supports subdirectories
    nested = write_file(tmp_path, "subdir/file.txt", "nested content")
```

### `tmp_artifacts`

Provides an isolated artifacts directory for tests. **Automatically patches `get_run_dir()` to use the temp directory.**

```python
def test_cli_output(tmp_artifacts):
    # CLI will write to tmp_artifacts/runs/ingest/test/...
    result = runner.invoke(app, ["--run-id", "test_run", ...])

    # Verify output in temp directory
    run_dir = get_run_dir("test_run", "ingest")
    assert (run_dir / "samples.ndjson").exists()

    # Cleanup happens automatically after test
```

**Benefits:**
- No pollution of real `artifacts/` directory
- Tests can run in parallel without conflicts
- Automatic cleanup via `tmp_path` fixture

### `mock_samples`

Creates a mock `samples.ndjson` file with 2 JudgingExample records:

```python
def test_infer(mock_samples):
    result = runner.invoke(app, ["--input", str(mock_samples), ...])
    assert result.exit_code == 0
```

### `mock_judgements`

Creates a mock `judgements.ndjson` file with 2 valid Judgement records:

```python
def test_schema_validation(mock_judgements):
    valid, invalid, errors = validate_ndjson_file(mock_judgements, "judgement")
    assert valid == 2
    assert invalid == 0
```

### `mock_llm_judge_dataset`

Creates a complete LLM Judge dataset structure (queries, documents, qrels):

```python
def test_ingest(mock_llm_judge_dataset):
    result = runner.invoke(app, ["--data-dir", str(mock_llm_judge_dataset), ...])
    assert result.exit_code == 0
```

### `thomas_prompt_template`

Session-scoped fixture that loads the thomas-et-al prompt template once:

```python
def test_prompt_building(thomas_prompt_template):
    result = build_instruction(
        thomas_prompt_template,
        query="test",
        page_text="content"
    )
    assert "test" in result
```

## Test Markers

Tests should be marked with one or more markers for categorization:

```python
@pytest.mark.unit
def test_parse_query():
    """Fast, isolated unit test."""
    ...

@pytest.mark.integration
def test_cli_integration():
    """Integration test involving CLI and file I/O."""
    ...

@pytest.mark.slow
def test_expensive_operation():
    """Long-running test (>1s)."""
    ...

@pytest.mark.requires_api
def test_openrouter_client():
    """Test requiring API credentials."""
    ...
```

**Available markers:**
- `unit` — Fast, isolated, no I/O (preferred)
- `integration` — Tests involving files, adapters, or CLI
- `slow` — Tests taking >1 second
- `requires_api` — Tests requiring API keys or external services

## Writing New Tests

### 1. Unit Tests (Preferred)

Test pure domain logic in isolation:

```python
@pytest.mark.unit
def test_query_parser():
    """Test query parsing logic."""
    query = parse_query("q1\tWhat is AI?")
    assert query.query_id == "q1"
    assert query.query_text == "What is AI?"
```

**Best practices:**
- No file I/O
- No network calls
- No `tmp_artifacts` needed
- Fast execution (<10ms)

### 2. Adapter Tests

Test I/O adapters with temporary files:

```python
@pytest.mark.unit
def test_read_queries(tmp_path, write_file):
    """Test query file reader."""
    qfile = write_file(tmp_path, "queries.txt", "q1\tQuery\n")
    queries = read_queries(qfile)
    assert len(queries) == 1
```

**Best practices:**
- Use `write_file` fixture for creating test files
- Use `tmp_path` for isolation
- Test error cases (malformed files, missing files)

### 3. CLI Integration Tests

Test end-to-end CLI behavior:

```python
@pytest.mark.integration
class TestIngestCLI:
    def test_basic_ingest(self, mock_llm_judge_dataset, tmp_artifacts):
        """Test CLI produces valid output."""
        result = runner.invoke(
            app,
            [
                "--dataset", "llm-judge-2024",
                "--data-dir", str(mock_llm_judge_dataset),
                "--run-id", "test_run",
            ]
        )

        assert result.exit_code == 0

        # Verify output file (uses tmp_artifacts via fixture)
        run_dir = get_run_dir("test_run", "ingest")
        output_file = run_dir / "samples.ndjson"
        assert output_file.exists()
```

**Best practices:**
- Use `tmp_artifacts` fixture for isolated artifact directories
- Use `mock_*` fixtures for test data
- Test both success and failure cases
- Verify output file contents and schema validation

## Testing Anti-Patterns

**❌ Don't do this:**

```python
# BAD: Writes to real artifacts/ directory
def test_cli():
    result = runner.invoke(app, ["--dataset", "llm-judge-2024"])
    # Output goes to artifacts/runs/... (pollutes repo)

# BAD: No cleanup
def test_files():
    with open("/tmp/test_file.txt", "w") as f:
        f.write("test")
    # File left behind after test

# BAD: Duplicate helper functions
def test_foo():
    def _write(path, content):  # Duplicated in every test file
        ...
```

**✅ Do this instead:**

```python
# GOOD: Uses tmp_artifacts fixture
def test_cli(tmp_artifacts):
    result = runner.invoke(app, ["--dataset", "llm-judge-2024"])
    # Output goes to tmp_artifacts/runs/... (auto-cleanup)

# GOOD: Uses write_file fixture
def test_files(tmp_path, write_file):
    test_file = write_file(tmp_path, "test.txt", "content")
    # Automatically cleaned up via tmp_path

# GOOD: Uses shared fixture
def test_foo(write_file):
    # Uses centralized write_file from conftest.py
    ...
```

## Configuration

Test configuration is in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
addopts = "-q"
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
```

**Common pytest options:**
- `-v` — Verbose output
- `-s` — Show print statements
- `-x` — Stop on first failure
- `--lf` — Run last failed tests only
- `--ff` — Run failures first
- `--cov` — Generate coverage report

## Continuous Integration

Tests run automatically on:
- Pre-commit hooks (fast unit tests only)
- Pull requests (all tests except `requires_api`)
- Main branch pushes (full test suite)