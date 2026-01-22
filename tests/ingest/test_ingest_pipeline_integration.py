"""Integration tests for IngestApplication with mock adapters.

This test module demonstrates end-to-end pipeline testing with:
- REAL domain logic: IngestRunFactory, NormalizedDataset processing
- MOCK infrastructure: file I/O (external dependencies)

These tests verify that the complete ingest pipeline works correctly
with actual production logic, while avoiding slow external calls.

These are NOT unit tests for adapters (those exist separately).
These verify that components integrate correctly within the pipeline.
"""

from __future__ import annotations
import pytest
from pathlib import Path

from llm_ensemble.ingest.application.ingest_application import IngestApplication
from llm_ensemble.ingest.domain.entities.normalized_dataset import NormalizedDataset

from tests.ingest.mocks import (
    MockInputAdapter,
    MockOutputAdapter,
)


@pytest.mark.integration
def test_ingest_pipeline_basic(
    sample_dataset: NormalizedDataset,
    temp_run_dir: Path
):
    """Test end-to-end ingest pipeline with mock adapters.

    This demonstrates the testing benefits of Ports & Adapters architecture:
    - REAL domain logic (IngestRunFactory, summary building)
    - MOCK infrastructure (file I/O)
    - Fast execution without external dependencies

    Tests application business logic:
    - Dataset is read and passed through pipeline
    - IngestRun entity is correctly built
    - Write summary tracks created entities
    - Run summary contains correct statistics
    """
    # Arrange: Mock adapters
    input_adapter = MockInputAdapter(sample_dataset)
    output_adapter = MockOutputAdapter()

    app = IngestApplication(
        reader=input_adapter,
        writer=output_adapter,
        run_dir=temp_run_dir,
        run_name="integration-test-run",
        io_name="mock",
    )

    # Act: Run ingest pipeline
    summary = app.run_ingest(
        input_path=Path("/mock/input/path"),
        limit=None,
        official=False,
        notes="Integration test run",
    )

    # Assert: Input adapter was called correctly
    assert input_adapter.read_called
    assert input_adapter.read_call_args["input_path"] == Path("/mock/input/path")
    assert input_adapter.read_call_args["limit"] is None

    # Assert: Output adapter received the ingest run
    assert output_adapter.written_ingest_run is not None

    # Assert: Summary has correct sample count
    assert summary.sample_count == 2

    # Assert: Write summary tracks created entities
    assert summary.write_summary.samples_created == 2
    assert summary.write_summary.datasets_created == 1
    assert summary.write_summary.configs_created == 1
    assert summary.write_summary.runs_created == 1

    # Assert: Run metadata is correct
    assert summary.run.run_name == "integration-test-run"
    assert summary.run.notes == "Integration test run"


@pytest.mark.integration
def test_ingest_pipeline_with_limit(
    sample_dataset_five: NormalizedDataset,
    temp_run_dir: Path
):
    """Test ingest pipeline with sample limit.

    Verifies that the limit parameter correctly restricts the number
    of samples processed through the pipeline.
    """
    # Arrange: Mock adapters
    input_adapter = MockInputAdapter(sample_dataset_five)
    output_adapter = MockOutputAdapter()

    app = IngestApplication(
        reader=input_adapter,
        writer=output_adapter,
        run_dir=temp_run_dir,
        run_name="limit-test-run",
        io_name="mock",
    )

    # Act: Run ingest with limit of 3 samples
    summary = app.run_ingest(
        input_path=Path("/mock/input/path"),
        limit=3,
        official=False,
        notes="Testing limit parameter",
    )

    # Assert: Input adapter was called with limit
    assert input_adapter.read_call_args["limit"] == 3

    # Assert: Only limited samples were processed
    assert summary.sample_count == 3
    assert summary.write_summary.samples_created == 3


@pytest.mark.integration
def test_ingest_pipeline_official_run(
    sample_dataset: NormalizedDataset,
    temp_run_dir: Path
):
    """Test ingest pipeline marks official runs correctly.

    Verifies that the official flag is properly propagated through
    the pipeline and reflected in the run entity.
    """
    # Arrange
    input_adapter = MockInputAdapter(sample_dataset)
    output_adapter = MockOutputAdapter()

    app = IngestApplication(
        reader=input_adapter,
        writer=output_adapter,
        run_dir=temp_run_dir,
        run_name="official-test-run",
        io_name="mock",
    )

    # Act: Run as official
    summary = app.run_ingest(
        input_path=Path("/mock/input/path"),
        limit=None,
        official=True,
        notes="Official run test",
    )

    # Assert: Run is marked as official
    from llm_ensemble.libs.runtime.run_info import RunType
    assert summary.run.run_type == RunType.OFFICIAL


@pytest.mark.integration
def test_ingest_pipeline_timing(
    sample_dataset: NormalizedDataset,
    temp_run_dir: Path
):
    """Test that ingest pipeline records timing information.

    Verifies that start and end times are captured and that
    end time is after start time.
    """
    # Arrange
    input_adapter = MockInputAdapter(sample_dataset)
    output_adapter = MockOutputAdapter()

    app = IngestApplication(
        reader=input_adapter,
        writer=output_adapter,
        run_dir=temp_run_dir,
        run_name="timing-test-run",
        io_name="mock",
    )

    # Act
    summary = app.run_ingest(
        input_path=Path("/mock/input/path"),
        limit=None,
        official=False,
        notes=None,
    )

    # Assert: Timing is recorded
    assert summary.start_time is not None
    assert summary.end_time is not None
    assert summary.end_time >= summary.start_time
