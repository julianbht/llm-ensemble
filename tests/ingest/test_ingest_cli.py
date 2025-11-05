"""Integration tests for the ingest CLI.

These tests verify end-to-end CLI behavior with real adapters and file I/O.
"""

import json
import pytest
from pathlib import Path
from typer.testing import CliRunner

from llm_ensemble.ingest_cli import app


runner = CliRunner()


@pytest.fixture
def sample_llm_judge_dataset(tmp_path: Path, write_file):
    """Create a minimal LLM Judge dataset for testing.

    Creates the 3 required files with 2 query-document pairs.
    """
    data_dir = tmp_path / "llm_judge_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Query file
    write_file(
        data_dir,
        "llm4eval_query_2024.txt",
        "q1\tWhat is Python?\nq2\tWhat is Java?\n"
    )

    # Document file
    write_file(
        data_dir,
        "llm4eval_document_2024.jsonl",
        json.dumps({"docid": "d1", "doc": "Python is a programming language."}) + "\n"
        + json.dumps({"docid": "d2", "doc": "Java is another language."}) + "\n",
    )

    # Qrel file
    write_file(
        data_dir,
        "llm4eval_test_qrel_2024.txt",
        "q1 2 d1\nq2 1 d2\n"
    )

    return data_dir


@pytest.mark.integration
class TestIngestCLIEndToEnd:
    """Test CLI end-to-end with real I/O."""

    def test_cli_ndjson_output(
        self,
        sample_llm_judge_dataset: Path,
        tmp_runs_dir,
        mock_git_info,
    ):
        """Test that CLI produces correct NDJSON output with full provenance."""
        artifacts_dir, get_run_dir = tmp_runs_dir

        # Invoke CLI
        result = runner.invoke(
            app,
            [
                "--input", str(sample_llm_judge_dataset),
                "--io-cfg", "llm_judge_challenge_ndjson",
                "--run-name", "ndjson_test",
            ],
        )

        # Verify CLI succeeded
        assert result.exit_code == 0, f"CLI failed with output:\n{result.stdout}\n{result.stderr}"

        # Verify expected log events in stderr
        assert "INGEST_STARTED" in result.stderr or "ingest_started" in result.stderr.lower()
        assert "INGEST_COMPLETE" in result.stderr or "ingest_complete" in result.stderr.lower()

        # Get run directory
        run_dir = get_run_dir("ingest", "ndjson_test", official=False)
        assert run_dir.exists(), f"Run directory not found: {run_dir}"

        # Verify output file exists
        output_file = run_dir / "normalized_dataset.ndjson"
        assert output_file.exists(), f"Output file not found: {output_file}"

        # Read and verify NDJSON content
        lines = [line for line in output_file.read_text().strip().split("\n") if line]
        assert len(lines) == 2, f"Expected 2 samples, got {len(lines)}"

        # Parse first record
        sample1 = json.loads(lines[0])

        # Verify sample structure
        assert "id" in sample1, "Missing 'id' field"
        assert "query" in sample1, "Missing 'query' field"
        assert "document" in sample1, "Missing 'document' field"
        assert "gold_score" in sample1, "Missing 'gold_score' field"
        assert "run_info" in sample1, "Missing 'run_info' field"

        # Verify run_info metadata
        run_info = sample1["run_info"]
        assert run_info["run_name"] == "ndjson_test"
        assert run_info["io_config_name"] == "llm_judge_challenge_ndjson"
        assert run_info["input_path"] == str(sample_llm_judge_dataset)

        # Verify git info from mock
        assert run_info["git_sha"] == "abc1234"
        assert run_info["git_clean"] is True
        assert run_info["git_branch"] == "test-branch"

        # Verify query and document content
        assert sample1["query"]["external_id"] == "q1"
        assert sample1["query"]["text"] == "What is Python?"
        assert sample1["document"]["external_id"] == "d1"
        assert "Python is a programming language" in sample1["document"]["text"]
        assert sample1["gold_score"]["score"] == 2

        # Verify summary.json exists and has correct counts
        summary_file = run_dir / "summary.json"
        assert summary_file.exists(), f"Summary file not found: {summary_file}"

        with open(summary_file) as f:
            summary = json.load(f)
            assert summary["sample_count"] == 2
            assert summary["write_summary"]["samples_created"] == 2

    def test_cli_with_limit(
        self,
        sample_llm_judge_dataset: Path,
        tmp_runs_dir,
        mock_git_info,
    ):
        """Test that --limit flag correctly constrains output."""
        artifacts_dir, get_run_dir = tmp_runs_dir

        # Invoke CLI with limit
        result = runner.invoke(
            app,
            [
                "--input", str(sample_llm_judge_dataset),
                "--io-cfg", "llm_judge_challenge_ndjson",
                "--run-name", "limit_test",
                "--limit", "1",
            ],
        )

        # Verify success
        assert result.exit_code == 0, f"CLI failed: {result.stdout}\n{result.stderr}"

        # Get run directory
        run_dir = get_run_dir("ingest", "limit_test", official=False)
        output_file = run_dir / "normalized_dataset.ndjson"

        # Verify only 1 sample was written
        lines = [line for line in output_file.read_text().strip().split("\n") if line]
        assert len(lines) == 1, f"Expected 1 sample with --limit 1, got {len(lines)}"

        # Verify summary reflects limit
        summary_file = run_dir / "summary.json"
        with open(summary_file) as f:
            summary = json.load(f)
            assert summary["sample_count"] == 1
            assert summary["run_info"]["limit"] == 1


@pytest.mark.integration
class TestIngestCLIOverrides:
    """Test CLI configuration overrides."""

    def test_cli_override_writer_module(
        self,
        sample_llm_judge_dataset: Path,
        tmp_runs_dir,
        mock_git_info,
    ):
        """Test that --override can change the writer adapter."""
        artifacts_dir, get_run_dir = tmp_runs_dir

        # Invoke CLI with overrides to use JSON writer instead of NDJSON
        result = runner.invoke(
            app,
            [
                "--input", str(sample_llm_judge_dataset),
                "--io-cfg", "llm_judge_challenge_ndjson",  # Base config uses NDJSON writer
                "--run-name", "override_writer_test",
                "--override", "io.writer_module=llm_ensemble.ingest.adapters.io.fully_populated_json_writer",
                "--override", "io.writer_class=FullyPopulatedJsonWriter",
            ],
        )

        # Verify success
        assert result.exit_code == 0, f"CLI failed: {result.stdout}\n{result.stderr}"

        # Get run directory
        run_dir = get_run_dir("ingest", "override_writer_test", official=False)

        # Verify NDJSON file does NOT exist (was overridden)
        ndjson_file = run_dir / "normalized_dataset.ndjson"
        assert not ndjson_file.exists(), "NDJSON file should not exist when writer is overridden"

        # Verify JSON file DOES exist (from override)
        json_file = run_dir / "normalized_dataset.json"
        assert json_file.exists(), f"JSON file should exist after override: {json_file}"

        # Verify JSON content
        with open(json_file) as f:
            data = json.load(f)
            # FullyPopulatedJsonWriter writes a JSON array
            assert isinstance(data, list), "Expected JSON array"
            assert len(data) == 2, f"Expected 2 samples, got {len(data)}"

            # Verify first sample structure
            sample = data[0]
            assert "id" in sample
            assert "query" in sample
            assert "document" in sample
            assert "gold_score" in sample
            assert "run_info" in sample

    def test_cli_override_dataset_name(
        self,
        sample_llm_judge_dataset: Path,
        tmp_runs_dir,
        mock_git_info,
    ):
        """Test that --override can change dataset_name in config."""
        artifacts_dir, get_run_dir = tmp_runs_dir

        custom_dataset_name = "custom-test-dataset"

        # Invoke CLI with dataset_name override
        result = runner.invoke(
            app,
            [
                "--input", str(sample_llm_judge_dataset),
                "--io-cfg", "llm_judge_challenge_ndjson",
                "--run-name", "override_dataset_test",
                "--override", f"io.dataset_name={custom_dataset_name}",
            ],
        )

        # Verify success
        assert result.exit_code == 0, f"CLI failed: {result.stdout}\n{result.stderr}"

        # Get run directory and read output
        run_dir = get_run_dir("ingest", "override_dataset_test", official=False)
        output_file = run_dir / "normalized_dataset.ndjson"

        # Parse first sample
        first_line = output_file.read_text().strip().split("\n")[0]
        sample = json.loads(first_line)

        # Verify dataset_name in run_info reflects override
        assert sample["run_info"]["io_config"]["dataset_name"] == custom_dataset_name

        # Verify summary.json also reflects override
        summary_file = run_dir / "summary.json"
        with open(summary_file) as f:
            summary = json.load(f)
            assert summary["run_info"]["io_config"]["dataset_name"] == custom_dataset_name

    def test_cli_multiple_overrides(
        self,
        sample_llm_judge_dataset: Path,
        tmp_runs_dir,
        mock_git_info,
    ):
        """Test that multiple --override flags work together."""
        artifacts_dir, get_run_dir = tmp_runs_dir

        # Invoke CLI with multiple overrides
        result = runner.invoke(
            app,
            [
                "--input", str(sample_llm_judge_dataset),
                "--io-cfg", "llm_judge_challenge_ndjson",
                "--run-name", "multi_override_test",
                "--override", "io.dataset_name=multi-test",
                "--override", "io.dataset_description=Testing multiple overrides",
                "--override", "io.writer_module=llm_ensemble.ingest.adapters.io.fully_populated_json_writer",
                "--override", "io.writer_class=FullyPopulatedJsonWriter",
            ],
        )

        # Verify success
        assert result.exit_code == 0, f"CLI failed: {result.stdout}\n{result.stderr}"

        # Get run directory
        run_dir = get_run_dir("ingest", "multi_override_test", official=False)

        # Verify JSON file exists (writer override worked)
        json_file = run_dir / "normalized_dataset.json"
        assert json_file.exists()

        # Verify dataset metadata overrides
        with open(json_file) as f:
            data = json.load(f)
            sample = data[0]
            assert sample["run_info"]["io_config"]["dataset_name"] == "multi-test"
            assert "multiple overrides" in sample["run_info"]["io_config"]["dataset_description"].lower()


@pytest.mark.integration
class TestIngestCLIErrorHandling:
    """Test CLI error handling."""

    def test_cli_fails_on_missing_input_dir(self, tmp_runs_dir, mock_git_info):
        """Test that CLI fails gracefully when input directory doesn't exist."""
        result = runner.invoke(
            app,
            [
                "ingest",
                "--input", "/nonexistent/directory",
                "--io-cfg", "llm_judge_challenge_ndjson",
                "--run-name", "error_test",
            ],
        )

        # Should fail with non-zero exit code
        assert result.exit_code != 0

        # Error message should be informative
        output = result.stdout + result.stderr
        assert "does not exist" in output.lower() or "not found" in output.lower()

    def test_cli_fails_on_invalid_io_config(self, tmp_path: Path, tmp_runs_dir, mock_git_info):
        """Test that CLI fails when invalid io-cfg is specified."""
        # Create empty input dir
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        result = runner.invoke(
            app,
            [
                "ingest",
                "--input", str(input_dir),
                "--io-cfg", "nonexistent_config",
                "--run-name", "error_test",
            ],
        )

        # Should fail
        assert result.exit_code != 0

        # Error should mention config not found
        output = result.stdout + result.stderr
        assert "config" in output.lower() and ("not found" in output.lower() or "no" in output.lower())
