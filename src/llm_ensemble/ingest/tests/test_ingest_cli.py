"""Integration tests for the ingest CLI.

These tests verify the CLI entrypoint produces correct NDJSON output
and handles edge cases properly.
"""

import json
from pathlib import Path
from typer.testing import CliRunner

from llm_ensemble.ingest.cli.ingest_cli import app


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

    def test_basic_ingest_to_stdout(self, tmp_path: Path):
        """Test that ingest writes valid NDJSON to stdout."""
        data_dir = _setup_basic_dataset(tmp_path)

        result = runner.invoke(
            app, ["--dataset", "llm-judge", "--data-dir", str(data_dir)]
        )

        assert result.exit_code == 0
        lines = result.stdout.strip().split("\n")
        # Last line is the summary to stderr, so check stdout
        assert "Wrote 1 examples" in result.stderr

        # Parse the NDJSON output
        example = json.loads(lines[0])
        assert example["query_id"] == "q1"
        assert example["docid"] == "d1"
        assert example["gold_relevance"] == 1

    def test_ingest_to_file(self, tmp_path: Path):
        """Test that ingest writes to a specified output file."""
        data_dir = _setup_basic_dataset(tmp_path / "data")
        out_file = tmp_path / "output" / "samples.ndjson"

        result = runner.invoke(
            app,
            [
                "--dataset",
                "llm-judge",
                "--data-dir",
                str(data_dir),
                "--out",
                str(out_file),
            ],
        )

        assert result.exit_code == 0
        assert out_file.exists()

        # Verify content
        content = out_file.read_text()
        example = json.loads(content.strip())
        assert example["query_id"] == "q1"

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

        result = runner.invoke(
            app,
            [
                "--dataset",
                "llm-judge",
                "--data-dir",
                str(tmp_path),
                "--limit",
                "2",
            ],
        )

        assert result.exit_code == 0
        assert "Wrote 2 examples" in result.stderr
        lines = [l for l in result.stdout.strip().split("\n") if l]
        assert len(lines) == 2

    def test_unsupported_dataset_fails(self, tmp_path: Path):
        """Test that unsupported dataset names are rejected."""
        result = runner.invoke(
            app,
            [
                "--dataset",
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
                "--dataset",
                "llm-judge",
                "--data-dir",
                "/nonexistent/path",
            ],
        )

        assert result.exit_code != 0

    def test_output_creates_parent_dirs(self, tmp_path: Path):
        """Test that output file creation creates parent directories."""
        data_dir = _setup_basic_dataset(tmp_path / "data")
        out_file = tmp_path / "deeply" / "nested" / "output.ndjson"

        result = runner.invoke(
            app,
            [
                "--dataset",
                "llm-judge",
                "--data-dir",
                str(data_dir),
                "--out",
                str(out_file),
            ],
        )

        assert result.exit_code == 0
        assert out_file.exists()
        assert out_file.parent.exists()

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

        result = runner.invoke(
            app,
            ["--dataset", "llm-judge", "--data-dir", str(tmp_path)],
        )

        assert result.exit_code == 0
        lines = [l for l in result.stdout.strip().split("\n") if l]
        assert len(lines) == 2

        # Each line should be valid JSON
        ex1 = json.loads(lines[0])
        ex2 = json.loads(lines[1])
        assert ex1["query_id"] == "q1"
        assert ex2["query_id"] == "q2"