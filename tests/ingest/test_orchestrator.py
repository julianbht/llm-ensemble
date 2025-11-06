"""Tests for ingest orchestrator (run_ingest function).

These tests verify the orchestration layer that handles infrastructure
concerns like run directories, logging, and adapter instantiation.
"""

import json
import pytest
import re
from pathlib import Path

from llm_ensemble.ingest.orchestrator import run_ingest
from llm_ensemble.ingest.schemas import IngestIOConfig, WriteSummary
from llm_ensemble.ingest.schemas.ingest_run_summary import IngestRunSummary
from llm_ensemble.libs.schemas import LoggingConfig
from llm_ensemble.libs.runtime.run_info import RunType


@pytest.fixture
def minimal_io_config():
    """Create a minimal IngestIOConfig for testing."""
    return IngestIOConfig(
        name_hint="test",
        description="Test IO config",
        dataset_name="test-dataset",
        dataset_description="Test dataset",
        reader_module="test.reader",
        reader_class="TestReader",
        writer_module="test.writer",
        writer_class="TestWriter",
    )


@pytest.fixture
def default_logging_config():
    """Create a default LoggingConfig for testing."""
    return LoggingConfig(
        pretty_print=False,
        save_logs=False,
        console_level="INFO",
        file_level="DEBUG",
    )


@pytest.mark.integration
class TestRunIngestOfficialFlow:
    """Test run_ingest orchestrator with official flow."""

    def test_official_run_creates_official_directory(
        self,
        tmp_path: Path,
        tmp_runs_dir,
        mock_git_info,
        minimal_io_config,
        default_logging_config,
        monkeypatch,
    ):
        """Test that official=True creates run in official/ subdirectory."""
        artifacts_dir, get_run_dir = tmp_runs_dir

        # Create fake input directory
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        # Create fake service that returns a summary
        from llm_ensemble.ingest.domain import IngestionService

        class FakeService:
            def __init__(self, sample_reader, dataset_writer):
                self.sample_reader = sample_reader
                self.dataset_writer = dataset_writer

            def ingest_dataset(self, data_dir, run_info, run_dir, limit=None, on_write=None):
                # Verify run_info is correct
                assert run_info.run_type == RunType.OFFICIAL
                assert run_info.io_config_name == "test_config"

                # Create a fake summary
                write_summary = WriteSummary(samples_created=5)
                if on_write:
                    on_write(write_summary)

                from llm_ensemble.libs.runtime.run_summary_builder import RunSummaryBuilder
                builder = RunSummaryBuilder(run_info)
                builder.set_start_time()
                builder.add("sample_count", 5)
                builder.add("write_summary", write_summary)
                return builder.finalize(IngestRunSummary)

        # Patch IngestionService
        monkeypatch.setattr("llm_ensemble.ingest.orchestrator.IngestionService", FakeService)

        # Patch the config's get_reader and get_writer methods
        # We need to patch at the class level since these are methods on IngestIOConfig
        monkeypatch.setattr(
            "llm_ensemble.ingest.schemas.ingest_io_config.IngestIOConfig.get_reader",
            lambda self: None
        )
        monkeypatch.setattr(
            "llm_ensemble.ingest.schemas.ingest_io_config.IngestIOConfig.get_writer",
            lambda self: None
        )

        # Run ingest with official=True and fixed run_name
        run_ingest(
            io_config=minimal_io_config,
            logging_config=default_logging_config,
            io_config_name="test_config",
            input_path=input_dir,
            run_name="fixed_test_run",
            limit=None,
            official=True,
            notes="Test official run",
        )

        # Verify run directory was created in official/ subdirectory
        expected_run_dir = get_run_dir("ingest", "fixed_test_run", official=True)
        assert expected_run_dir.exists()
        assert "official" in str(expected_run_dir)

        # Verify summary.json was written
        summary_file = expected_run_dir / "summary.json"
        assert summary_file.exists()

        # Verify summary content
        with open(summary_file) as f:
            summary = json.load(f)
            assert summary["sample_count"] == 5
            assert summary["run_info"]["run_type"] == "official"
