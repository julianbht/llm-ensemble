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

        # Verify output files exist (samples + manifest)
        samples_file = run_dir / "judging_samples.ndjson"
        manifest_file = run_dir / "ingest_run_info.json"
        assert samples_file.exists(), f"Samples file not found: {samples_file}"
        assert manifest_file.exists(), f"Manifest file not found: {manifest_file}"

        # Read and verify NDJSON content (samples should NOT have run_info embedded)
        lines = [line for line in samples_file.read_text().strip().split("\n") if line]
        assert len(lines) == 2, f"Expected 2 samples, got {len(lines)}"

        # Parse first record
        sample1 = json.loads(lines[0])

        # Verify sample structure (pure domain entity without run_info)
        assert "id" in sample1, "Missing 'id' field"
        assert "query" in sample1, "Missing 'query' field"
        assert "document" in sample1, "Missing 'document' field"
        assert "gold_score" in sample1, "Missing 'gold_score' field"
        assert "run_info" not in sample1, "run_info should not be embedded in samples"

        # Verify run_info is in separate manifest file
        with open(manifest_file) as f:
            run_info = json.load(f)
            assert run_info["run_name"] == "ndjson_test"
            assert run_info["io_config_name"] == "llm_judge_challenge_ndjson"
            assert run_info["input_path"] == str(sample_llm_judge_dataset)

            # Verify git info from mock (or actual git if not mocked)
            assert "git_sha" in run_info
            assert "git_clean" in run_info
            assert "git_branch" in run_info

        # Verify query and document content
        assert sample1["query"]["external_id"] == "q1"
        assert sample1["query"]["query_text"] == "What is Python?"
        assert sample1["document"]["external_id"] == "d1"
        assert "Python is a programming language" in sample1["document"]["doc_text"]
        assert sample1["gold_score"] == 2

        # Verify summary.json exists and has correct counts
        summary_file = run_dir / "summary.json"
        assert summary_file.exists(), f"Summary file not found: {summary_file}"

        with open(summary_file) as f:
            summary = json.load(f)
            assert summary["sample_count"] == 2
            assert summary["write_summary"]["samples_created"] == 2

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
                "--override", "io.writer_module=llm_ensemble.ingest.adapters.io.fully_populated_json_writer",
                "--override", "io.writer_class=FullyPopulatedJsonWriter",
            ],
        )

        # Verify success
        assert result.exit_code == 0, f"CLI failed: {result.stdout}\n{result.stderr}"

        # Get run directory
        run_dir = get_run_dir("ingest", "multi_override_test", official=False)

        # Verify JSON files exist (writer override worked)
        json_file = run_dir / "judging_samples.json"
        manifest_file = run_dir / "ingest_run_info.json"
        assert json_file.exists(), f"Samples file not found: {json_file}"
        assert manifest_file.exists(), f"Manifest file not found: {manifest_file}"

        # Verify writer module override worked (JSON writer instead of NDJSON)
        with open(manifest_file) as f:
            run_info = json.load(f)
            assert run_info["io_config"]["writer_module"] == "llm_ensemble.ingest.adapters.io.fully_populated_json_writer"
            assert run_info["io_config"]["writer_class"] == "FullyPopulatedJsonWriter"

        # Verify samples are clean domain entities without run_info
        with open(json_file) as f:
            data = json.load(f)
            sample = data[0]
            assert "run_info" not in sample, "run_info should not be embedded in samples"


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

        # Error message should be informative (typer shows error in various ways)
        output = result.stdout + result.stderr
        assert ("does not exist" in output.lower() or 
                "not found" in output.lower() or 
                "error" in output.lower())
