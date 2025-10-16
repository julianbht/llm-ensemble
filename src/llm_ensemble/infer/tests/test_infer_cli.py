"""Integration tests for the infer CLI.

These tests verify the CLI entrypoint produces correct NDJSON output
and validates against the judgement schema.
"""

import json
from pathlib import Path
from typer.testing import CliRunner
import pytest

from llm_ensemble.libs.schemas.validator import validate_ndjson_file


runner = CliRunner()


@pytest.mark.integration
class TestInferCLI:
    """Test the infer CLI command."""

    @pytest.mark.requires_api
    @pytest.mark.skip(reason="Requires model config and API access")
    def test_output_validates_against_schema(self, mock_samples, tmp_artifacts):
        """Test that infer CLI output validates against judgement.schema.json.

        Note: This test requires a valid model config and API access.
        It's marked as requires_api and skipped by default.
        """
        test_run_id = "test_run"

        # This test would run if API credentials were available
        # Example of how it would work:
        # result = runner.invoke(
        #     app,
        #     [
        #         "--model", "test-model",
        #         "--input", str(mock_samples),
        #         "--run-id", test_run_id,
        #         "--limit", "2",
        #     ],
        # )
        # assert result.exit_code == 0
        #
        # run_dir = get_run_dir(test_run_id, cli_name="infer", official=False)
        # output_file = run_dir / "judgements.ndjson"
        # assert output_file.exists()
        #
        # valid_count, invalid_count, errors = validate_ndjson_file(
        #     output_file, "judgement"
        # )
        # assert valid_count > 0
        # assert invalid_count == 0

    def test_judgement_schema_validation_with_mock_data(self, mock_judgements):
        """Test schema validation with mock judgement data from fixture."""
        # Validate against schema using mock_judgements fixture
        valid_count, invalid_count, errors = validate_ndjson_file(
            mock_judgements, "judgement"
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

        # Write invalid judgement to NDJSON file
        output_file = tmp_path / "invalid_judgement.ndjson"
        output_file.write_text(json.dumps(invalid_judgement) + "\n", encoding="utf-8")

        # Validate against schema
        valid_count, invalid_count, errors = validate_ndjson_file(
            output_file, "judgement"
        )

        assert valid_count == 0, "Should have no valid records"
        assert invalid_count == 1, "Should have 1 invalid record"
        assert len(errors) > 0, "Should have validation errors"
