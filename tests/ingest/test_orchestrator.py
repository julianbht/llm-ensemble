"""Tests for ingest orchestrator (run_ingest function).

These tests verify the orchestration layer that handles infrastructure
concerns like run directories, logging, and adapter instantiation.
"""

import json
import pytest
from pathlib import Path

from llm_ensemble.ingest.orchestrator import run_ingest
from llm_ensemble.ingest.schemas import WriteSummary
from llm_ensemble.ingest.schemas.ingest_run_summary import IngestRunSummary
from llm_ensemble.libs.schemas import LoggingConfig, IOConfig
from llm_ensemble.libs.runtime.run_info import RunType


@pytest.fixture
def minimal_io_config():
    """Create a minimal IOConfig for testing."""
    return IOConfig(
        name_hint="test",
        description="Test IO config",
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
            def __init__(self, dataset_reader, dataset_writer):
                self.dataset_reader = dataset_reader
                self.dataset_writer = dataset_writer

            def ingest_dataset(self, data_dir, run_info, limit=None, on_read_complete=None, on_write=None):
                # Verify run_info is correct
                assert run_info.run_type == RunType.OFFICIAL
                assert run_info.io_config_name == "test_config"

                # Create a fake summary
                write_summary = WriteSummary(samples_created=5)
                if on_write:
                    on_write(write_summary)

                from llm_ensemble.libs.runtime.run_summary_builder import RunSummaryBuilder
                builder = RunSummaryBuilder()
                builder.set_start_time()
                builder.add("sample_count", 5)
                builder.add("write_summary", write_summary)
                return builder.finalize(IngestRunSummary)

        # Patch IngestionService
        monkeypatch.setattr("llm_ensemble.ingest.orchestrator.IngestionService", FakeService)

        # Patch the config's get_reader and get_writer methods
        # We need to patch at the class level since these are methods on IOConfig
        monkeypatch.setattr(
            "llm_ensemble.libs.schemas.io_config.IOConfig.get_reader",
            lambda self: None
        )
        monkeypatch.setattr(
            "llm_ensemble.libs.schemas.io_config.IOConfig.get_writer",
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

        # Verify summary content (run_info is now persisted separately)
        with open(summary_file) as f:
            summary = json.load(f)
            assert summary["sample_count"] == 5
            assert "run_info" not in summary  # run_info is persisted separately now

    def test_run_metadata_completeness(
        self,
        tmp_path: Path,
        tmp_runs_dir,
        mock_git_info,
        minimal_io_config,
        default_logging_config,
        monkeypatch,
    ):
        """Test that all metadata (git info, notes, timestamps, config) are captured correctly."""
        artifacts_dir, get_run_dir = tmp_runs_dir

        # Create fake input directory
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        # Track run_info passed to service
        captured_run_info = None

        # Create fake service
        from llm_ensemble.ingest.domain import IngestionService

        class FakeService:
            def __init__(self, dataset_reader, dataset_writer):
                pass

            def ingest_dataset(self, data_dir, run_info, limit=None, on_read_complete=None, on_write=None):
                nonlocal captured_run_info
                captured_run_info = run_info

                # Create summary
                write_summary = WriteSummary(samples_created=10)
                if on_write:
                    on_write(write_summary)

                from llm_ensemble.libs.runtime.run_summary_builder import RunSummaryBuilder
                builder = RunSummaryBuilder()
                builder.set_start_time()
                builder.add("sample_count", 10)
                builder.add("write_summary", write_summary)
                return builder.finalize(IngestRunSummary)

        monkeypatch.setattr("llm_ensemble.ingest.orchestrator.IngestionService", FakeService)

        # Patch config methods
        monkeypatch.setattr(
            "llm_ensemble.libs.schemas.io_config.IOConfig.get_reader",
            lambda self: None
        )
        monkeypatch.setattr(
            "llm_ensemble.libs.schemas.io_config.IOConfig.get_writer",
            lambda self: None
        )

        # Run with all metadata parameters
        test_notes = "Comprehensive metadata test run"
        run_ingest(
            io_config=minimal_io_config,
            logging_config=default_logging_config,
            io_config_name="test_config",
            input_path=input_dir,
            run_name="metadata_test_run",
            limit=100,
            official=True,
            notes=test_notes,
        )

        # Verify run_info has all required metadata
        assert captured_run_info is not None
        
        # Basic identification
        assert captured_run_info.run_name == "metadata_test_run"
        assert captured_run_info.cli_name == "ingest"
        assert captured_run_info.run_type == RunType.OFFICIAL
        
        # User-provided metadata
        assert captured_run_info.notes == test_notes
        assert captured_run_info.limit == 100
        
        # Git metadata (should be present, actual values may vary)
        assert captured_run_info.git_sha is not None
        assert len(captured_run_info.git_sha) > 0
        assert captured_run_info.git_clean is not None
        assert captured_run_info.git_branch is not None
        assert len(captured_run_info.git_branch) > 0
        
        # Configuration metadata
        assert captured_run_info.io_config_name == "test_config"
        assert captured_run_info.input_path == str(input_dir)
        
        # Verify summary.json has aggregate metrics (no run_info)
        # Note: run_info is now persisted separately by writers (ingest_run_info.json)
        # This test uses a fake service, so we only verify summary.json content
        run_dir = get_run_dir("ingest", "metadata_test_run", official=True)
        summary_file = run_dir / "summary.json"

        with open(summary_file) as f:
            summary = json.load(f)
            assert "sample_count" in summary
            assert summary["sample_count"] == 10
            assert "run_info" not in summary  # run_info is persisted separately by writers now

        # The captured_run_info already verifies all metadata was passed correctly to the domain service
        # Writers are responsible for persisting run_info to ingest_run_info.json (tested separately)
            
            # Verify timing metadata exists
            assert "start_time" in summary
            assert "end_time" in summary
            
            # Verify sample count is correct
            assert summary["sample_count"] == 10
