"""Tests for IngestionService domain logic.

These tests verify the pure business logic of the ingestion pipeline
in complete isolation from infrastructure concerns.
"""

import pytest
from pathlib import Path
from uuid import UUID

from llm_ensemble.ingest.domain import IngestionService
from llm_ensemble.ingest.schemas import (
    Query, Document, RelevanceScore, IngestRunInfo, Dataset
)
from llm_ensemble.libs.schemas import IOConfig
from llm_ensemble.libs.runtime.run_info import RunType


@pytest.fixture
def sample_io_config():
    """Create a minimal IOConfig for testing."""
    return IOConfig(
        name_hint="test",
        description="Test config",
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
def sample_normalized_dataset():
    """Create sample NormalizedDataset for testing."""
    from llm_ensemble.ingest.schemas import NormalizedDataset, JudgingSample

    # Create a dataset entity
    dataset = Dataset.create("test-dataset", "Test dataset for domain service")

    samples = [
        JudgingSample.create(
            query=Query.create(dataset, "q1", "What is Python?"),
            document=Document.create(dataset, "d1", "Python is a programming language."),
            gold_score=RelevanceScore.HIGHLY_RELEVANT,  # 2
        ),
        JudgingSample.create(
            query=Query.create(dataset, "q2", "What is Java?"),
            document=Document.create(dataset, "d2", "Java is another programming language."),
            gold_score=RelevanceScore.RELEVANT,  # 1
        ),
        JudgingSample.create(
            query=Query.create(dataset, "q3", "What is C++?"),
            document=Document.create(dataset, "d3", "C++ is a compiled language."),
            gold_score=RelevanceScore.IRRELEVANT,  # 0
        ),
    ]

    return NormalizedDataset(dataset=dataset, samples=samples)


@pytest.mark.unit
class TestIngestionServiceLimitAndCallbacks:
    """Test IngestionService limit and callback functionality."""

    def test_limit_constrains_samples(
        self,
        tmp_path: Path,
        fake_reader_factory,
        fake_writer_factory,
        sample_normalized_dataset,
        sample_run_info,
    ):
        """Test that limit parameter constrains the number of samples processed."""
        # Create fake reader with normalized dataset (3 samples)
        reader = fake_reader_factory(sample_normalized_dataset)
        writer = fake_writer_factory()

        # Create service
        service = IngestionService(dataset_reader=reader, dataset_writer=writer)

        # Run with limit=2
        summary = service.ingest_dataset(
            data_dir=tmp_path,
            run_info=sample_run_info,
            limit=2,
        )

        # Verify reader received limit
        assert reader.called_with["limit"] == 2

        # Verify writer received exactly 2 samples
        assert len(writer.written_samples) == 2
        assert summary.sample_count == 2
        assert summary.write_summary.samples_created == 2
