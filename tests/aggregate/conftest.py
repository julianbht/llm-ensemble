"""Pytest fixtures for aggregate CLI tests.

This module provides shared test fixtures that are automatically discovered
by pytest and available to all test files in the tests/aggregate/ directory.
"""

from __future__ import annotations
import pytest
from pathlib import Path
from uuid import uuid4

from llm_ensemble.ingest.domain.entities.normalized_dataset import NormalizedDataset
from llm_ensemble.ingest.domain.entities.dataset_sample import (
    NormalizedDatasetJudgingSample,
)
from llm_ensemble.ingest.domain.entities.judging_sample import JudgingSample
from llm_ensemble.ingest.domain.entities.query import Query
from llm_ensemble.ingest.domain.entities.document import Document
from llm_ensemble.infer.domain.entities.infer_run_output import InferRunOutput
from llm_ensemble.infer.domain.entities.llm_judgement import (
    LLMJudgement,
    LLMInvocationMetrics,
)
from llm_ensemble.infer.domain.entities.llm_score import LLMScore
from llm_ensemble.infer.domain.entities.llm_prompt_text import LLMPromptText
from llm_ensemble.infer.domain.entities.llm_response_text import LLMResponseText
from llm_ensemble.libs.schemas.relevance_score import RelevanceScore


def _create_llm_judgement(
    dataset_sample: NormalizedDatasetJudgingSample,
    label: RelevanceScore,
) -> LLMJudgement:
    """Helper to create an LLMJudgement for testing."""
    return LLMJudgement(
        dataset_sample=dataset_sample,
        llm_prompt_text=LLMPromptText(prompt_text="Test prompt"),
        llm_response_text=LLMResponseText(llm_response_text='{"M": 1}'),
        llm_invocation_metrics=LLMInvocationMetrics(
            latency_ms=100.0,
            retries=0,
            cost_estimate_usd=0.001,
            actual_cost_usd=None,
            generation_id=None,
            prompt_tokens=50,
            completion_tokens=20,
            total_tokens=70,
        ),
        llm_score=LLMScore(label=label, confidence=None, rationale=None),
        parser_issue=None,
    )


@pytest.fixture
def sample_dataset_three() -> NormalizedDataset:
    """Create a test dataset with three samples.

    Returns:
        NormalizedDataset with 3 judging samples
    """
    dataset_id = uuid4()

    samples = [
        NormalizedDatasetJudgingSample(
            normalized_dataset_id=dataset_id,
            judging_sample=JudgingSample(
                query=Query(query_text="What is Python?"),
                document=Document(doc_text="Python is a programming language."),
                gold_score=RelevanceScore.RELEVANT,
            ),
            sequence_number=0,
        ),
        NormalizedDatasetJudgingSample(
            normalized_dataset_id=dataset_id,
            judging_sample=JudgingSample(
                query=Query(query_text="How to install packages?"),
                document=Document(doc_text="Use pip install to add Python packages."),
                gold_score=RelevanceScore.HIGHLY_RELEVANT,
            ),
            sequence_number=1,
        ),
        NormalizedDatasetJudgingSample(
            normalized_dataset_id=dataset_id,
            judging_sample=JudgingSample(
                query=Query(query_text="What is machine learning?"),
                document=Document(
                    doc_text="Machine learning is a subset of artificial intelligence."
                ),
                gold_score=RelevanceScore.RELEVANT,
            ),
            sequence_number=2,
        ),
    ]

    return NormalizedDataset(
        id=dataset_id,
        fingerprint="mock-fingerprint-aggregate",
        external_dataset_name="test-dataset-aggregate",
        samples=samples,
    )


@pytest.fixture
def infer_run_outputs_three_models(
    sample_dataset_three: NormalizedDataset,
) -> list[InferRunOutput]:
    """Create three InferRunOutputs simulating three different models.

    Each model judged the same samples (same fingerprint).
    Votes are designed to test majority voting:
    - Sample 0: votes [1, 1, 2] -> majority = 1
    - Sample 1: votes [2, 2, 2] -> unanimous = 2
    - Sample 2: votes [0, 1, 1] -> majority = 1

    Returns:
        List of 3 InferRunOutput objects with matching fingerprints
    """
    fingerprint = sample_dataset_three.fingerprint
    samples = sample_dataset_three.samples

    # Model 1 votes: [1, 2, 0]
    model1_judgements = [
        _create_llm_judgement(samples[0], RelevanceScore.RELEVANT),
        _create_llm_judgement(samples[1], RelevanceScore.HIGHLY_RELEVANT),
        _create_llm_judgement(samples[2], RelevanceScore.IRRELEVANT),
    ]
    output1 = InferRunOutput(
        llm_judgements=model1_judgements,
        sample_fingerprint=fingerprint,
        finished=True,
        judgement_count=3,
        failed_parses_count=0,
        avg_latency_ms=100.0,
    )

    # Model 2 votes: [1, 2, 1]
    model2_judgements = [
        _create_llm_judgement(samples[0], RelevanceScore.RELEVANT),
        _create_llm_judgement(samples[1], RelevanceScore.HIGHLY_RELEVANT),
        _create_llm_judgement(samples[2], RelevanceScore.RELEVANT),
    ]
    output2 = InferRunOutput(
        llm_judgements=model2_judgements,
        sample_fingerprint=fingerprint,
        finished=True,
        judgement_count=3,
        failed_parses_count=0,
        avg_latency_ms=120.0,
    )

    # Model 3 votes: [2, 2, 1]
    model3_judgements = [
        _create_llm_judgement(samples[0], RelevanceScore.HIGHLY_RELEVANT),
        _create_llm_judgement(samples[1], RelevanceScore.HIGHLY_RELEVANT),
        _create_llm_judgement(samples[2], RelevanceScore.RELEVANT),
    ]
    output3 = InferRunOutput(
        llm_judgements=model3_judgements,
        sample_fingerprint=fingerprint,
        finished=True,
        judgement_count=3,
        failed_parses_count=0,
        avg_latency_ms=110.0,
    )

    return [output1, output2, output3]


@pytest.fixture
def temp_run_dir(tmp_path: Path) -> Path:
    """Create temporary run directory for testing.

    Uses pytest's built-in tmp_path fixture which automatically creates
    and cleans up temporary directories.

    Returns:
        Path to temporary directory (auto-cleaned by pytest)
    """
    return tmp_path
