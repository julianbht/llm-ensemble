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

    def test_auto_generated_run_name_includes_timestamp(
        self,
        tmp_path: Path,
        tmp_runs_dir,
        mock_git_info,
        minimal_io_config,
        default_logging_config,
        monkeypatch,
    ):
        """Test that run_name is auto-generated with timestamp when not provided."""
        artifacts_dir, get_run_dir = tmp_runs_dir

        # Create fake input directory
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        # Track generated run_name
        captured_run_name = None

        # Create fake service
        from llm_ensemble.ingest.domain import IngestionService

        class FakeService:
            def __init__(self, sample_reader, dataset_writer):
                pass

            def ingest_dataset(self, data_dir, run_info, run_dir, limit=None, on_write=None):
                nonlocal captured_run_name
                captured_run_name = run_info.run_name

                # Create minimal summary
                write_summary = WriteSummary(samples_created=1)
                if on_write:
                    on_write(write_summary)

                from llm_ensemble.libs.runtime.run_summary_builder import RunSummaryBuilder
                builder = RunSummaryBuilder(run_info)
                builder.set_start_time()
                builder.add("sample_count", 1)
                builder.add("write_summary", write_summary)
                return builder.finalize(IngestRunSummary)

        monkeypatch.setattr("llm_ensemble.ingest.orchestrator.IngestionService", FakeService)

        # Patch config methods at class level
        monkeypatch.setattr(
            "llm_ensemble.ingest.schemas.ingest_io_config.IngestIOConfig.get_reader",
            lambda self: None
        )
        monkeypatch.setattr(
            "llm_ensemble.ingest.schemas.ingest_io_config.IngestIOConfig.get_writer",
            lambda self: None
        )

        # Run without specifying run_name
        run_ingest(
            io_config=minimal_io_config,
            logging_config=default_logging_config,
            io_config_name="test_config",
            input_path=input_dir,
            run_name=None,  # Auto-generate
            limit=None,
            official=False,
            notes=None,
        )

        # Verify run_name was generated (should match timestamp pattern)
        assert captured_run_name is not None
        # Pattern: YYYYMMDD_HHMMSS_hint or similar
        assert re.match(r"\d{8}_\d{6}", captured_run_name), \
            f"Expected timestamp pattern in run_name, got: {captured_run_name}"

    def test_run_info_contains_git_metadata(
        self,
        tmp_path: Path,
        tmp_runs_dir,
        mock_git_info,
        minimal_io_config,
        default_logging_config,
        monkeypatch,
    ):
        """Test that run_info contains git SHA, branch, and clean status."""
        artifacts_dir, get_run_dir = tmp_runs_dir

        # Create fake input directory
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        # Track run_info
        captured_run_info = None

        # Create fake service
        from llm_ensemble.ingest.domain import IngestionService

        class FakeService:
            def __init__(self, sample_reader, dataset_writer):
                pass

            def ingest_dataset(self, data_dir, run_info, run_dir, limit=None, on_write=None):
                nonlocal captured_run_info
                captured_run_info = run_info

                # Create minimal summary
                write_summary = WriteSummary(samples_created=1)
                if on_write:
                    on_write(write_summary)

                from llm_ensemble.libs.runtime.run_summary_builder import RunSummaryBuilder
                builder = RunSummaryBuilder(run_info)
                builder.set_start_time()
                builder.add("sample_count", 1)
                builder.add("write_summary", write_summary)
                return builder.finalize(IngestRunSummary)

        monkeypatch.setattr("llm_ensemble.ingest.orchestrator.IngestionService", FakeService)

        # Patch config methods at class level
        monkeypatch.setattr(
            "llm_ensemble.ingest.schemas.ingest_io_config.IngestIOConfig.get_reader",
            lambda self: None
        )
        monkeypatch.setattr(
            "llm_ensemble.ingest.schemas.ingest_io_config.IngestIOConfig.get_writer",
            lambda self: None
        )

        # Run ingest
        run_ingest(
            io_config=minimal_io_config,
            logging_config=default_logging_config,
            io_config_name="test_config",
            input_path=input_dir,
            run_name="test_git_run",
            limit=None,
            official=False,
            notes=None,
        )

        # Verify git info from mock_git_info fixture
        assert captured_run_info is not None
        assert captured_run_info.git_sha == "abc1234"
        assert captured_run_info.git_clean is True
        assert captured_run_info.git_branch == "test-branch"

    def test_notes_propagated_to_run_info(
        self,
        tmp_path: Path,
        tmp_runs_dir,
        mock_git_info,
        minimal_io_config,
        default_logging_config,
        monkeypatch,
    ):
        """Test that notes are propagated to run_info."""
        artifacts_dir, get_run_dir = tmp_runs_dir

        # Create fake input directory
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        # Track run_info
        captured_run_info = None

        # Create fake service
        from llm_ensemble.ingest.domain import IngestionService

        class FakeService:
            def __init__(self, sample_reader, dataset_writer):
                pass

            def ingest_dataset(self, data_dir, run_info, run_dir, limit=None, on_write=None):
                nonlocal captured_run_info
                captured_run_info = run_info

                # Create minimal summary
                write_summary = WriteSummary(samples_created=1)
                if on_write:
                    on_write(write_summary)

                from llm_ensemble.libs.runtime.run_summary_builder import RunSummaryBuilder
                builder = RunSummaryBuilder(run_info)
                builder.set_start_time()
                builder.add("sample_count", 1)
                builder.add("write_summary", write_summary)
                return builder.finalize(IngestRunSummary)

        monkeypatch.setattr("llm_ensemble.ingest.orchestrator.IngestionService", FakeService)

        # Patch config methods at class level
        monkeypatch.setattr(
            "llm_ensemble.ingest.schemas.ingest_io_config.IngestIOConfig.get_reader",
            lambda self: None
        )
        monkeypatch.setattr(
            "llm_ensemble.ingest.schemas.ingest_io_config.IngestIOConfig.get_writer",
            lambda self: None
        )

        # Run with notes
        test_notes = "Testing notes propagation"
        run_ingest(
            io_config=minimal_io_config,
            logging_config=default_logging_config,
            io_config_name="test_config",
            input_path=input_dir,
            run_name="test_notes_run",
            limit=None,
            official=False,
            notes=test_notes,
        )

        # Verify notes in run_info
        assert captured_run_info is not None
        assert captured_run_info.notes == test_notes

    def test_summary_json_written_to_run_dir(
        self,
        tmp_path: Path,
        tmp_runs_dir,
        mock_git_info,
        minimal_io_config,
        default_logging_config,
        monkeypatch,
    ):
        """Test that summary.json is written to the run directory."""
        artifacts_dir, get_run_dir = tmp_runs_dir

        # Create fake input directory
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        # Create fake service
        from llm_ensemble.ingest.domain import IngestionService

        class FakeService:
            def __init__(self, sample_reader, dataset_writer):
                pass

            def ingest_dataset(self, data_dir, run_info, run_dir, limit=None, on_write=None):
                # Create summary with test data
                write_summary = WriteSummary(samples_created=42)
                if on_write:
                    on_write(write_summary)

                from llm_ensemble.libs.runtime.run_summary_builder import RunSummaryBuilder
                builder = RunSummaryBuilder(run_info)
                builder.set_start_time()
                builder.add("sample_count", 42)
                builder.add("write_summary", write_summary)
                return builder.finalize(IngestRunSummary)

        monkeypatch.setattr("llm_ensemble.ingest.orchestrator.IngestionService", FakeService)

        # Patch config methods at class level
        monkeypatch.setattr(
            "llm_ensemble.ingest.schemas.ingest_io_config.IngestIOConfig.get_reader",
            lambda self: None
        )
        monkeypatch.setattr(
            "llm_ensemble.ingest.schemas.ingest_io_config.IngestIOConfig.get_writer",
            lambda self: None
        )

        # Run ingest
        run_ingest(
            io_config=minimal_io_config,
            logging_config=default_logging_config,
            io_config_name="test_config",
            input_path=input_dir,
            run_name="test_summary_run",
            limit=None,
            official=False,
            notes=None,
        )

        # Verify summary.json exists
        run_dir = get_run_dir("ingest", "test_summary_run", official=False)
        summary_file = run_dir / "summary.json"
        assert summary_file.exists()

        # Verify content
        with open(summary_file) as f:
            summary = json.load(f)
            assert summary["sample_count"] == 42
            assert summary["write_summary"]["samples_created"] == 42
