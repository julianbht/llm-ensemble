"""Integration tests for the infer CLI.

These tests verify the CLI entrypoint produces correct NDJSON output
and validates against the judgement schema.
"""

import json
from pathlib import Path
from typer.testing import CliRunner
import pytest

from llm_ensemble.infer_cli import app
from llm_ensemble.libs.schemas.validator import validate_ndjson_file
from llm_ensemble.libs.runtime.run_manager import get_run_dir


runner = CliRunner()


def _write(tmp: Path, name: str, content: str) -> Path:
    """Helper to write test fixture files."""
    p = tmp / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def _setup_mock_input_samples(tmp_path: Path) -> Path:
    """Create mock JudgingExample samples for inference."""
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

    tmp_path.mkdir(parents=True, exist_ok=True)
    samples_file = tmp_path / "samples.ndjson"
    with open(samples_file, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    return samples_file


class TestInferCLI:
    """Test the infer CLI command."""

    def test_output_validates_against_schema(self, tmp_path: Path):
        """Test that infer CLI output validates against judgement.schema.json.

        Note: This test requires a valid model config and API access.
        It's marked as integration and may be skipped in CI without credentials.
        """
        # Setup input samples
        input_file = _setup_mock_input_samples(tmp_path / "input")

        # Get run directory using run_manager (matches CLI behavior)
        test_run_id = "test_run"
        run_dir = get_run_dir(test_run_id, cli_name="infer", official=False)
        output_file = run_dir / "judgements.ndjson"

        # This test requires a working model config and API access
        # Skip if model config doesn't exist or API keys aren't set
        pytest.skip("Integration test - requires model config and API access")

        # If we had a mock model, the test would look like:
        # result = runner.invoke(
        #     app,
        #     [
        #         "--model", "test-model",
        #         "--input", str(input_file),
        #         "--run-id", "test_run",
        #         "--limit", "2",
        #     ],
        # )
        #
        # assert result.exit_code == 0
        # assert output_file.exists()
        #
        # # Validate against schema
        # valid_count, invalid_count, errors = validate_ndjson_file(
        #     output_file, "judgement"
        # )
        #
        # assert valid_count > 0, "No valid records found"
        # assert invalid_count == 0, f"Schema validation failed: {errors}"

    def test_judgement_schema_validation_with_mock_data(self, tmp_path: Path):
        """Test schema validation with manually created mock judgement data."""
        # Create a mock judgement file that should validate
        mock_judgements = [
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

        # Write mock judgements
        output_file = tmp_path / "mock_judgements.ndjson"
        with open(output_file, "w", encoding="utf-8") as f:
            for judgement in mock_judgements:
                f.write(json.dumps(judgement) + "\n")

        # Validate against schema
        valid_count, invalid_count, errors = validate_ndjson_file(
            output_file, "judgement"
        )

        assert valid_count == 2, "Expected 2 valid records"
        assert invalid_count == 0, f"Schema validation failed: {errors}"
        assert len(errors) == 0, f"Unexpected errors: {errors}"

    def test_judgement_schema_catches_invalid_label(self, tmp_path: Path):
        """Test that schema validation catches invalid label values."""
        invalid_judgement = {
            "model_id": "test-model",
            "provider": "openrouter",
            "query_id": "q1",
            "docid": "d1",
            "label": 5,  # Invalid - should be 0, 1, 2, or null
            "raw_text": "Invalid label",
            "latency_ms": 100.0,
            "warnings": [],
        }

        output_file = tmp_path / "invalid_judgement.ndjson"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(invalid_judgement) + "\n")

        # Validate against schema
        valid_count, invalid_count, errors = validate_ndjson_file(
            output_file, "judgement"
        )

        assert valid_count == 0, "Should have no valid records"
        assert invalid_count == 1, "Should have 1 invalid record"
        assert len(errors) > 0, "Should have validation errors"
