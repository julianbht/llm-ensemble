"""Tests for IngestionService domain logic.

These tests verify the pure business logic of the ingestion pipeline
in complete isolation from infrastructure concerns.
"""

import pytest
from pathlib import Path
from uuid import UUID

from llm_ensemble.ingest.domain import IngestionService
from llm_ensemble.ingest.schemas import (
    Query, Document, RelevanceScore, IngestIOConfig, IngestRunInfo, Dataset
)
from llm_ensemble.ingest.ports.sample_reader import RawJudgingSample
from llm_ensemble.libs.runtime.run_info import RunType


@pytest.fixture
def sample_io_config():
    """Create a minimal IngestIOConfig for testing."""
    return IngestIOConfig(
        name_hint="test",
        description="Test config",
        dataset_name="test-dataset",
        dataset_description="Test dataset description",
        reader_module="test.reader",
        reader_class="TestReader",
        writer_module="test.writer",
        writer_class="TestWriter",
    )


@pytest.fixture
def sample_run_info(sample_io_config):
    """Create a minimal IngestRunInfo for testing."""
    return IngestRunInfo.create(
        run_name="test_run",
        io_config_name="test_config",
        io_config=sample_io_config,
        input_path="/test/path",
        limit=None,
        run_type=RunType.TEST,
        notes=None,
        git_sha="abc123",
        git_clean=True,
        git_branch="test-branch",
    )


@pytest.fixture
def sample_raw_samples():
    """Create sample RawJudgingSample objects for testing."""
    # Create a dataset entity
    dataset = Dataset.create("test-dataset", "Test dataset for domain service")

    return [
        RawJudgingSample(
            query=Query.create(dataset, "q1", "What is Python?"),
            document=Document.create(dataset, "d1", "Python is a programming language."),
            gold_score=RelevanceScore.HIGHLY_RELEVANT,  # 2
        ),
        RawJudgingSample(
            query=Query.create(dataset, "q2", "What is Java?"),
            document=Document.create(dataset, "d2", "Java is another programming language."),
            gold_score=RelevanceScore.RELEVANT,  # 1
        ),
        RawJudgingSample(
            query=Query.create(dataset, "q3", "What is C++?"),
            document=Document.create(dataset, "d3", "C++ is a compiled language."),
            gold_score=RelevanceScore.IRRELEVANT,  # 0
        ),
    ]


@pytest.mark.unit
class TestIngestionServiceLimitAndCallbacks:
    """Test IngestionService limit and callback functionality."""

    def test_limit_constrains_samples(
        self,
        tmp_path: Path,
        fake_reader_factory,
        fake_writer_factory,
        sample_raw_samples,
        sample_run_info,
    ):
        """Test that limit parameter constrains the number of samples processed."""
        # Create fake reader with 3 samples
        reader = fake_reader_factory(sample_raw_samples)
        writer = fake_writer_factory()

        # Create service
        service = IngestionService(sample_reader=reader, dataset_writer=writer)

        # Run with limit=2
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        summary = service.ingest_dataset(
            data_dir=tmp_path,
            run_info=sample_run_info,
            run_dir=run_dir,
            limit=2,
        )

        # Verify reader received limit
        assert reader.called_with["limit"] == 2

        # Verify writer received exactly 2 samples
        assert len(writer.written_samples) == 2
        assert summary.sample_count == 2
        assert summary.write_summary.samples_created == 2

    def test_no_limit_processes_all_samples(
        self,
        tmp_path: Path,
        fake_reader_factory,
        fake_writer_factory,
        sample_raw_samples,
        sample_run_info,
    ):
        """Test that without limit, all samples are processed."""
        # Create fake reader with 3 samples
        reader = fake_reader_factory(sample_raw_samples)
        writer = fake_writer_factory()

        # Create service
        service = IngestionService(sample_reader=reader, dataset_writer=writer)

        # Run without limit
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        summary = service.ingest_dataset(
            data_dir=tmp_path,
            run_info=sample_run_info,
            run_dir=run_dir,
            limit=None,
        )

        # Verify reader received None for limit
        assert reader.called_with["limit"] is None

        # Verify writer received all 3 samples
        assert len(writer.written_samples) == 3
        assert summary.sample_count == 3
        assert summary.write_summary.samples_created == 3

    def test_on_sample_callback_invoked_for_each_sample(
        self,
        tmp_path: Path,
        fake_reader_factory,
        fake_writer_factory,
        sample_raw_samples,
        sample_run_info,
    ):
        """Test that on_sample callback is invoked for each processed sample."""
        # Create fake reader and writer
        reader = fake_reader_factory(sample_raw_samples)
        writer = fake_writer_factory()

        # Create service
        service = IngestionService(sample_reader=reader, dataset_writer=writer)

        # Track callback invocations
        callback_samples = []

        def on_sample(sample):
            callback_samples.append(sample)

        # Run with callback
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        service.ingest_dataset(
            data_dir=tmp_path,
            run_info=sample_run_info,
            run_dir=run_dir,
            limit=None,
            on_sample=on_sample,
        )

        # Verify callback was called 3 times
        assert len(callback_samples) == 3

        # Verify samples have run_info attached
        for sample in callback_samples:
            assert sample.run_info == sample_run_info
            assert isinstance(sample.id, UUID)

    def test_on_write_callback_invoked_after_write(
        self,
        tmp_path: Path,
        fake_reader_factory,
        fake_writer_factory,
        sample_raw_samples,
        sample_run_info,
    ):
        """Test that on_write callback is invoked after write completes."""
        # Create fake reader and writer
        reader = fake_reader_factory(sample_raw_samples)
        writer = fake_writer_factory()

        # Create service
        service = IngestionService(sample_reader=reader, dataset_writer=writer)

        # Track callback invocations
        write_summaries = []

        def on_write(write_summary):
            write_summaries.append(write_summary)

        # Run with callback
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        service.ingest_dataset(
            data_dir=tmp_path,
            run_info=sample_run_info,
            run_dir=run_dir,
            limit=None,
            on_write=on_write,
        )

        # Verify callback was called once
        assert len(write_summaries) == 1

        # Verify summary has correct counts
        summary = write_summaries[0]
        assert summary.samples_created == 3
        assert summary.total_created == 3

    def test_callbacks_invoked_in_correct_order(
        self,
        tmp_path: Path,
        fake_reader_factory,
        fake_writer_factory,
        sample_raw_samples,
        sample_run_info,
    ):
        """Test that callbacks are invoked in the correct order."""
        # Create fake reader and writer
        reader = fake_reader_factory(sample_raw_samples)
        writer = fake_writer_factory()

        # Create service
        service = IngestionService(sample_reader=reader, dataset_writer=writer)

        # Track callback order
        call_order = []

        def on_sample(sample):
            call_order.append(f"sample:{sample.query.external_id}")

        def on_write(write_summary):
            call_order.append(f"write:{write_summary.samples_created}")

        # Run with callbacks
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        service.ingest_dataset(
            data_dir=tmp_path,
            run_info=sample_run_info,
            run_dir=run_dir,
            limit=2,
            on_sample=on_sample,
            on_write=on_write,
        )

        # Verify order: all samples, then write
        assert call_order == ["sample:q1", "sample:q2", "write:2"]

    def test_run_info_attached_to_all_samples(
        self,
        tmp_path: Path,
        fake_reader_factory,
        fake_writer_factory,
        sample_raw_samples,
        sample_run_info,
    ):
        """Test that run_info is attached to all processed samples."""
        # Create fake reader and writer
        reader = fake_reader_factory(sample_raw_samples)
        writer = fake_writer_factory()

        # Create service
        service = IngestionService(sample_reader=reader, dataset_writer=writer)

        # Run ingestion
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        service.ingest_dataset(
            data_dir=tmp_path,
            run_info=sample_run_info,
            run_dir=run_dir,
            limit=None,
        )

        # Verify all written samples have run_info
        for sample in writer.written_samples:
            assert sample.run_info == sample_run_info
            assert sample.run_info.run_name == "test_run"
            assert sample.run_info.io_config_name == "test_config"

    def test_summary_contains_timing_info(
        self,
        tmp_path: Path,
        fake_reader_factory,
        fake_writer_factory,
        sample_raw_samples,
        sample_run_info,
    ):
        """Test that returned summary contains timing information."""
        # Create fake reader and writer
        reader = fake_reader_factory(sample_raw_samples)
        writer = fake_writer_factory()

        # Create service
        service = IngestionService(sample_reader=reader, dataset_writer=writer)

        # Run ingestion
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        summary = service.ingest_dataset(
            data_dir=tmp_path,
            run_info=sample_run_info,
            run_dir=run_dir,
            limit=None,
        )

        # Verify summary has timing info
        assert summary.start_time is not None
        assert summary.end_time is not None
        assert summary.end_time >= summary.start_time
