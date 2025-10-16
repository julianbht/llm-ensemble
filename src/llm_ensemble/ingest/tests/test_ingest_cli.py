"""Integration tests for the ingest CLI.

These tests verify the CLI entrypoint produces correct NDJSON output
and handles edge cases properly.
"""

import json
from pathlib import Path
from typer.testing import CliRunner
import pytest

from llm_ensemble.ingest_cli import app
from llm_ensemble.libs.schemas.validator import validate_ndjson_file
from llm_ensemble.libs.runtime.run_manager import get_run_dir


runner = CliRunner()


@pytest.mark.integration
class TestIngestCLI:
    """Test the ingest CLI command."""

    def test_limit_flag(self, tmp_path: Path, write_file, tmp_artifacts):
        """Test that --limit restricts the number of examples processed."""
        # Setup dataset with 3 examples
        write_file(
            tmp_path,
            "llm4eval_query_2024.txt",
            "q1\tQuery 1\nq2\tQuery 2\nq3\tQuery 3\n",
        )
        write_file(
            tmp_path,
            "llm4eval_document_2024.jsonl",
            json.dumps({"docid": "d1", "doc": "Doc 1"}) + "\n"
            + json.dumps({"docid": "d2", "doc": "Doc 2"}) + "\n"
            + json.dumps({"docid": "d3", "doc": "Doc 3"}) + "\n",
        )
        write_file(
            tmp_path,
            "llm4eval_test_qrel_2024.txt",
            "q1 1 d1\nq2 0 d2\nq3 1 d3\n",
        )

        test_run_id = "test_limit"
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

        # Verify output file has exactly 2 examples (uses tmp_artifacts via fixture)
        run_dir = get_run_dir(test_run_id, cli_name="ingest", official=False)
        output_file = run_dir / "samples.ndjson"
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

    def test_multiple_examples_ndjson_format(self, tmp_path: Path, write_file, tmp_artifacts):
        """Test that multiple examples are written as proper NDJSON (one per line)."""
        write_file(
            tmp_path,
            "llm4eval_query_2024.txt",
            "q1\tQuery 1\nq2\tQuery 2\n",
        )
        write_file(
            tmp_path,
            "llm4eval_document_2024.jsonl",
            json.dumps({"docid": "d1", "doc": "Doc 1"}) + "\n"
            + json.dumps({"docid": "d2", "doc": "Doc 2"}) + "\n",
        )
        write_file(
            tmp_path, "llm4eval_test_qrel_2024.txt", "q1 1 d1\nq2 0 d2\n"
        )

        test_run_id = "test_multiple"
        result = runner.invoke(
            app,
            ["--adapter", "llm-judge", "--data-dir", str(tmp_path), "--run-id", test_run_id],
        )

        assert result.exit_code == 0, f"CLI failed: {result.stderr}"

        # Read output file (uses tmp_artifacts via fixture)
        run_dir = get_run_dir(test_run_id, cli_name="ingest", official=False)
        output_file = run_dir / "samples.ndjson"
        assert output_file.exists()

        lines = [l for l in output_file.read_text().strip().split("\n") if l]
        assert len(lines) == 2

        # Each line should be valid JSON
        ex1 = json.loads(lines[0])
        ex2 = json.loads(lines[1])
        assert ex1["query_id"] == "q1"
        assert ex2["query_id"] == "q2"

    def test_output_validates_against_schema(self, mock_llm_judge_dataset, tmp_artifacts):
        """Test that ingest CLI output validates against sample.schema.json."""
        test_run_id = "test_schema_validation"

        result = runner.invoke(
            app,
            [
                "--adapter", "llm-judge",
                "--data-dir", str(mock_llm_judge_dataset),
                "--run-id", test_run_id,
            ],
        )

        assert result.exit_code == 0, f"CLI failed: {result.stderr}"

        # Get output file path using run_manager (uses tmp_artifacts via fixture)
        run_dir = get_run_dir(test_run_id, cli_name="ingest", official=False)
        output_file = run_dir / "samples.ndjson"
        assert output_file.exists(), f"Output file not found at {output_file}"

        # Validate against schema
        valid_count, invalid_count, errors = validate_ndjson_file(
            output_file, "sample"
        )

        assert valid_count > 0, "No valid records found"
        assert invalid_count == 0, f"Schema validation failed: {errors}"
        assert len(errors) == 0, f"Unexpected errors: {errors}"