"""Integration tests for InferenceApplication with mocked adapters.

This test module demonstrates the Ports & Adapters architecture's key benefit:
testing business logic in complete isolation from infrastructure.

The InferenceApplication orchestrates the inference pipeline by coordinating
interactions between five driven ports. By replacing real adapters with test
doubles (mocks), we can verify the application's orchestration logic without:
- Making actual LLM API calls
- Reading/writing files or databases
- Depending on external infrastructure

This enables fast, reliable, deterministic unit tests of complex business logic.

Architecture:
- Mock adapters defined in: tests/infer/mocks.py (reusable across tests)
- Test fixtures defined in: tests/infer/conftest.py (auto-discovered by pytest)
- Test logic defined here: assertions and verification
"""

from __future__ import annotations
import pytest
from pathlib import Path

from llm_ensemble.infer.application.inference_application import InferenceApplication
from llm_ensemble.ingest.domain.entities.normalized_dataset import NormalizedDataset
from llm_ensemble.libs.schemas.relevance_score import RelevanceScore

from tests.infer.mocks import (
    MockInputAdapter,
    MockOutputAdapter,
    MockPromptBuilder,
    MockLLMProvider,
    MockResponseParser,
)


@pytest.mark.integration
def test_inference_application_processes_all_samples_and_builds_summary(
    sample_dataset: NormalizedDataset,
    temp_run_dir: Path
):
    """Test that InferenceApplication processes dataset and computes run summary correctly.

    This test demonstrates the key benefit of Ports & Adapters architecture:
    the business logic (InferenceApplication) can be tested in complete isolation
    from infrastructure by injecting mock adapters.

    Focuses on testing actual business logic:
    - All samples are processed into judgements
    - Run summary aggregates metrics correctly (total latency, average, costs, etc.)
    - Judgements contain all required pipeline artifacts
    - Persistence counts are tracked

    Does NOT test mock internals - we trust that ports work if results are correct.
    """
    # Arrange: Create mock adapters
    output_adapter = MockOutputAdapter()
    app = InferenceApplication(
        input_port=MockInputAdapter(sample_dataset),
        output_port=output_adapter,
        prompt_builder=MockPromptBuilder(),
        llm_provider=MockLLMProvider(mock_response='{"M": 2, "T": 1, "O": 1}'),
        response_parser=MockResponseParser(),
        run_dir=temp_run_dir,
        run_name="test-run"
    )

    # Act: Run inference pipeline
    summary = app.run_inference(
        input_run_name="test-ingest-run",
        start_idx=None,
        end_idx=None,
        official=False,
        notes="Integration test run"
    )

    # Assert: All samples were processed into judgements
    assert len(output_adapter.written_judgements) == 2
    assert summary.judgements.total_count == 2

    # Assert: Run summary metrics are calculated correctly (BUSINESS LOGIC!)
    assert summary.judgements.failed_parses_count == 0
    assert summary.performance.latency.total_ms == 200.0  # 100ms * 2 samples
    assert summary.performance.latency.avg_ms == 100.0
    assert summary.performance.cost.total_usd == 0.002  # 0.001 * 2 samples
    assert summary.performance.tokens.total == 140  # 70 tokens * 2 samples
    assert summary.performance.tokens.total_prompt == 100  # 50 * 2
    assert summary.performance.tokens.total_completion == 40  # 20 * 2
    assert summary.persistence.total_created == 2

    # Assert: Judgements contain complete data from pipeline
    first_judgement = output_adapter.written_judgements[0]
    assert first_judgement.dataset_sample == sample_dataset.samples[0]
    assert first_judgement.llm_score is not None
    assert first_judgement.llm_score.label == RelevanceScore.RELEVANT
    assert first_judgement.llm_prompt_text is not None
    assert first_judgement.llm_response_text is not None
    assert first_judgement.llm_invocation_metrics is not None

    # Assert: Second sample was also processed
    second_judgement = output_adapter.written_judgements[1]
    assert second_judgement.dataset_sample == sample_dataset.samples[1]


@pytest.mark.integration
def test_inference_application_handles_parse_failures_gracefully(
    sample_dataset: NormalizedDataset,
    temp_run_dir: Path
):
    """Test that application handles parser failures without crashing the pipeline.

    Business logic under test:
    - Pipeline continues processing all samples even when parsing fails
    - Failed parses are counted correctly in run summary
    - Judgements are still created (with llm_score=None) for observability
    """
    # Arrange: Parser that simulates complete parse failure
    output_adapter = MockOutputAdapter()
    app = InferenceApplication(
        input_port=MockInputAdapter(sample_dataset),
        output_port=output_adapter,
        prompt_builder=MockPromptBuilder(),
        llm_provider=MockLLMProvider(),
        response_parser=MockResponseParser(mock_score=None),
        run_dir=temp_run_dir,
        run_name="test-run"
    )

    # Act: Run inference
    summary = app.run_inference(
        input_run_name="test-run",
        start_idx=None,
        end_idx=None,
        official=False,
        notes=None
    )

    # Assert: Pipeline processed all samples despite failures
    assert summary.judgements.total_count == 2
    assert len(output_adapter.written_judgements) == 2

    # Assert: Failures are tracked correctly (BUSINESS LOGIC!)
    assert summary.judgements.failed_parses_count == 2

    # Assert: Judgements exist but have no scores (for observability)
    assert output_adapter.written_judgements[0].llm_score is None
    assert output_adapter.written_judgements[1].llm_score is None


@pytest.mark.integration
def test_inference_application_processes_dataset_slice_correctly(
    sample_dataset: NormalizedDataset,
    temp_run_dir: Path
):
    """Test that application processes only the requested dataset slice.

    Business logic under test:
    - Only samples within [start_idx, end_idx) are processed
    - Run summary reflects slice size, not full dataset size
    - Correct samples are selected from dataset
    """
    # Arrange: Create application with mocks
    output_adapter = MockOutputAdapter()
    app = InferenceApplication(
        input_port=MockInputAdapter(sample_dataset),
        output_port=output_adapter,
        prompt_builder=MockPromptBuilder(),
        llm_provider=MockLLMProvider(),
        response_parser=MockResponseParser(),
        run_dir=temp_run_dir,
        run_name="test-run"
    )

    # Act: Process only first sample (slice [0:1])
    summary = app.run_inference(
        input_run_name="test-run",
        start_idx=0,
        end_idx=1,
        official=False,
        notes=None
    )

    # Assert: Only sliced samples were processed (BUSINESS LOGIC!)
    assert summary.judgements.total_count == 1
    assert len(output_adapter.written_judgements) == 1

    # Assert: Correct sample from dataset was selected
    assert output_adapter.written_judgements[0].dataset_sample.sequence_number == 0
    assert output_adapter.written_judgements[0].dataset_sample == sample_dataset.samples[0]


@pytest.mark.integration
def test_inference_application_manages_resource_lifecycle_correctly(
    sample_dataset: NormalizedDataset,
    temp_run_dir: Path
):
    """Test that application manages output port resource lifecycle properly.

    Business logic under test:
    - Output port is opened before writing (context manager protocol)
    - InferRun metadata is provided to output port at open time
    - Output port is closed after all writes complete
    - Write summary is available after closing

    This ensures proper resource management (file handles, DB connections, etc.)
    """
    # Arrange
    output_adapter = MockOutputAdapter()
    app = InferenceApplication(
        input_port=MockInputAdapter(sample_dataset),
        output_port=output_adapter,
        prompt_builder=MockPromptBuilder(),
        llm_provider=MockLLMProvider(),
        response_parser=MockResponseParser(),
        run_dir=temp_run_dir,
        run_name="test-run"
    )

    # Act
    app.run_inference(
        input_run_name="test-run",
        start_idx=None,
        end_idx=None,
        official=False,
        notes=None
    )

    # Assert: Output port received InferRun metadata (BUSINESS LOGIC!)
    assert output_adapter.infer_run is not None
    assert output_adapter.infer_run.run_name == "test-run"
    assert output_adapter.infer_run.run_type.value == "test"
    assert output_adapter.infer_run.infer_run_config is not None

    # Assert: Output port was properly closed (resource cleanup)
    assert not output_adapter.is_open

    # Assert: Write summary is available and correct
    write_summary = output_adapter.get_write_summary()
    assert write_summary.total_created == 2
    assert write_summary.llm_judgements_created == 2
