"""Integration tests for the ingest CLI.

These tests verify the CLI entrypoint produces correct NDJSON output
and handles edge cases properly.
"""

import json
from pathlib import Path
from typer.testing import CliRunner

from llm_ensemble.ingest_cli import app
from llm_ensemble.libs.schemas.validator import validate_ndjson_file


runner = CliRunner()


def _write(tmp: Path, name: str, content: str) -> Path:
    """Helper to write test fixture files."""
    p = tmp / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def _setup_basic_dataset(tmp_path: Path) -> Path:
    """Create a minimal valid dataset in tmp_path."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    _write(tmp_path, "llm4eval_query_2024.txt", "q1\tWhat is AI?\n")
    _write(
        tmp_path,
        "llm4eval_document_2024.jsonl",
        json.dumps({"docid": "d1", "doc": "AI is..."}) + "\n",
    )
    _write(tmp_path, "llm4eval_test_qrel_2024.txt", "q1 1 d1\n")
    return tmp_path


class TestIngestCLI:
    """Test the ingest CLI command."""

    def test_limit_flag(self, tmp_path: Path):
        """Test that --limit restricts the number of examples processed."""
        # Setup dataset with 3 examples
        _write(
            tmp_path,
            "llm4eval_query_2024.txt",
            "q1\tQuery 1\nq2\tQuery 2\nq3\tQuery 3\n",
        )
        _write(
            tmp_path,
            "llm4eval_document_2024.jsonl",
            json.dumps({"docid": "d1", "doc": "Doc 1"}) + "\n"
            + json.dumps({"docid": "d2", "doc": "Doc 2"}) + "\n"
            + json.dumps({"docid": "d3", "doc": "Doc 3"}) + "\n",
        )
        _write(
            tmp_path,
            "llm4eval_test_qrel_2024.txt",
            "q1 1 d1\nq2 0 d2\nq3 1 d3\n",
        )

        test_run_id = f"test_limit_{id(tmp_path)}"
        result = runner.invoke(
            app,
            [
                "--adapter",
                "llm-judge",
                "--data-dir",
                str(tmp_path),
                "--limit",
                "2",
                "--run-id",
                test_run_id,
            ],
        )

        assert result.exit_code == 0, f"CLI failed: {result.stderr}"
        assert "total_examples=2" in result.stderr

        # Verify output file has exactly 2 examples (now in test/ subdirectory)
        output_file = Path("artifacts") / "runs" / "ingest" / "test" / test_run_id / "samples.ndjson"
        assert output_file.exists()

        lines = [l for l in output_file.read_text().strip().split("\n") if l]
        assert len(lines) == 2

    def test_unsupported_dataset_fails(self, tmp_path: Path):
        """Test that unsupported dataset names are rejected."""
        result = runner.invoke(
            app,
            [
                "--adapter",
                "unknown-dataset",
                "--data-dir",
                str(tmp_path),
            ],
        )

        assert result.exit_code != 0
        # Typer shows errors in stdout
        output = result.stdout + result.stderr
        assert "Unsupported dataset" in output

    def test_missing_data_dir_fails(self):
        """Test that missing data directory is rejected."""
        result = runner.invoke(
            app,
            [
                "--adapter",
                "llm-judge",
                "--data-dir",
                "/nonexistent/path",
            ],
        )

        assert result.exit_code != 0

    def test_multiple_examples_ndjson_format(self, tmp_path: Path):
        """Test that multiple examples are written as proper NDJSON (one per line)."""
        _write(
            tmp_path,
            "llm4eval_query_2024.txt",
            "q1\tQuery 1\nq2\tQuery 2\n",
        )
        _write(
            tmp_path,
            "llm4eval_document_2024.jsonl",
            json.dumps({"docid": "d1", "doc": "Doc 1"}) + "\n"
            + json.dumps({"docid": "d2", "doc": "Doc 2"}) + "\n",
        )
        _write(
            tmp_path, "llm4eval_test_qrel_2024.txt", "q1 1 d1\nq2 0 d2\n"
        )

        test_run_id = f"test_multiple_{id(tmp_path)}"
        result = runner.invoke(
            app,
            ["--adapter", "llm-judge", "--data-dir", str(tmp_path), "--run-id", test_run_id],
        )

        assert result.exit_code == 0, f"CLI failed: {result.stderr}"

        # Read output file (now in test/ subdirectory)
        output_file = Path("artifacts") / "runs" / "ingest" / "test" / test_run_id / "samples.ndjson"
        assert output_file.exists()

        lines = [l for l in output_file.read_text().strip().split("\n") if l]
        assert len(lines) == 2

        # Each line should be valid JSON
        ex1 = json.loads(lines[0])
        ex2 = json.loads(lines[1])
        assert ex1["query_id"] == "q1"
        assert ex2["query_id"] == "q2"

    def test_output_validates_against_schema(self, tmp_path: Path):
        """Test that ingest CLI output validates against sample.schema.json."""
        data_dir = _setup_basic_dataset(tmp_path / "data")

        # Use a unique run_id for this test
        test_run_id = f"test_schema_validation_{id(tmp_path)}"

        result = runner.invoke(
            app,
            [
                "--adapter", "llm-judge",
                "--data-dir", str(data_dir),
                "--run-id", test_run_id,
            ],
        )

        assert result.exit_code == 0, f"CLI failed: {result.stderr}"

        # The CLI writes to artifacts/runs/ingest/test/<run_id>/ relative to project root
        # Find the output file
        output_file = Path("artifacts") / "runs" / "ingest" / "test" / test_run_id / "samples.ndjson"
        assert output_file.exists(), f"Output file not found at {output_file}"

        # Validate against schema
        valid_count, invalid_count, errors = validate_ndjson_file(
            output_file, "sample"
        )

        assert valid_count > 0, "No valid records found"
        assert invalid_count == 0, f"Schema validation failed: {errors}"
        assert len(errors) == 0, f"Unexpected errors: {errors}"