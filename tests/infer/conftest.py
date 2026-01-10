i"""Pytest fixtures for infer CLI tests.

This module provides shared test fixtures that are automatically discovered
by pytest and available to all test files in the tests/infer/ directory.
"""

from __future__ import annotations
import pytest
from pathlib import Path
from uuid import uuid4

from llm_ensemble.ingest.domain.entities.normalized_dataset import NormalizedDataset
from llm_ensemble.ingest.domain.entities.dataset_sample import NormalizedDatasetJudgingSample
from llm_ensemble.ingest.domain.entities.judging_sample import JudgingSample
from llm_ensemble.ingest.domain.entities.query import Query
from llm_ensemble.ingest.domain.entities.document import Document
from llm_ensemble.libs.schemas.relevance_score import RelevanceScore


@pytest.fixture
def sample_dataset() -> NormalizedDataset:
    """Create a small test dataset with two samples.

    This fixture provides a minimal but representative dataset for testing
    the inference pipeline without requiring real data files.

    Returns:
        NormalizedDataset with 2 judging samples covering different relevance scores
    """
    dataset_id = uuid4()

    # Sample 1: Relevant query-document pair
    sample1 = NormalizedDatasetJudgingSample(
        normalized_dataset_id=dataset_id,
        judging_sample=JudgingSample(
            query=Query(query_text="What is Python?"),
            document=Document(doc_text="Python is a programming language."),
            gold_score=RelevanceScore.RELEVANT
        ),
        sequence_number=0
    )

    # Sample 2: Highly relevant query-document pair
    sample2 = NormalizedDatasetJudgingSample(
        normalized_dataset_id=dataset_id,
        judging_sample=JudgingSample(
            query=Query(query_text="How to install packages?"),
            document=Document(doc_text="Use pip install to add Python packages."),
            gold_score=RelevanceScore.HIGHLY_RELEVANT
        ),
        sequence_number=1
    )

    return NormalizedDataset(
        id=dataset_id,
        fingerprint="mock-fingerprint-123",
        external_dataset_name="test-dataset",
        samples=[sample1, sample2]
    )


@pytest.fixture
def sample_dataset_five() -> NormalizedDataset:
    """Create a test dataset with five samples for slicing tests.

    Returns:
        NormalizedDataset with 5 judging samples with varied queries
    """
    dataset_id = uuid4()

    samples = [
        NormalizedDatasetJudgingSample(
            normalized_dataset_id=dataset_id,
            judging_sample=JudgingSample(
                query=Query(query_text="What is Python?"),
                document=Document(doc_text="Python is a programming language."),
                gold_score=RelevanceScore.RELEVANT
            ),
            sequence_number=0
        ),
        NormalizedDatasetJudgingSample(
            normalized_dataset_id=dataset_id,
            judging_sample=JudgingSample(
                query=Query(query_text="How to install packages?"),
                document=Document(doc_text="Use pip install to add Python packages."),
                gold_score=RelevanceScore.HIGHLY_RELEVANT
            ),
            sequence_number=1
        ),
        NormalizedDatasetJudgingSample(
            normalized_dataset_id=dataset_id,
            judging_sample=JudgingSample(
                query=Query(query_text="What is machine learning?"),
                document=Document(doc_text="Machine learning is a subset of artificial intelligence."),
                gold_score=RelevanceScore.RELEVANT
            ),
            sequence_number=2
        ),
        NormalizedDatasetJudgingSample(
            normalized_dataset_id=dataset_id,
            judging_sample=JudgingSample(
                query=Query(query_text="Python data structures"),
                document=Document(doc_text="Python has built-in data structures like lists and dictionaries."),
                gold_score=RelevanceScore.HIGHLY_RELEVANT
            ),
            sequence_number=3
        ),
        NormalizedDatasetJudgingSample(
            normalized_dataset_id=dataset_id,
            judging_sample=JudgingSample(
                query=Query(query_text="Best IDE for Python"),
                document=Document(doc_text="Popular Python IDEs include PyCharm and VS Code."),
                gold_score=RelevanceScore.RELEVANT
            ),
            sequence_number=4
        ),
    ]

    return NormalizedDataset(
        id=dataset_id,
        fingerprint="mock-fingerprint-456",
        external_dataset_name="test-dataset-five",
        samples=samples
    )


@pytest.fixture
def temp_run_dir(tmp_path: Path) -> Path:
    """Create temporary run directory for testing.

    Uses pytest's built-in tmp_path fixture which automatically creates
    and cleans up temporary directories.

    Returns:
        Path to temporary directory (auto-cleaned by pytest)
    """
    return tmp_path
