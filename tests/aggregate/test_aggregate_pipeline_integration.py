"""Integration tests for AggregationApplication with real domain adapters.

This test module demonstrates end-to-end pipeline testing with:
- REAL domain logic: actual aggregation strategies (business rules)
- MOCK infrastructure: file I/O (external dependencies)

These tests verify that the complete aggregation pipeline works correctly
with actual production adapters, while avoiding slow external calls.

Difference from unit tests:
- Unit tests: Test individual components in isolation
- This file: Tests real pipeline integration (hybrid mocks + real adapters)

These verify that real components integrate correctly within the pipeline.
"""

from __future__ import annotations
import pytest
from pathlib import Path

from llm_ensemble.aggregate.application.aggregation_application import AggregationApplication
from llm_ensemble.aggregate.adapters.driven.strategies.majority_vote_adapter import MajorityVoteAverage
from llm_ensemble.infer.domain.entities.infer_run_output import InferRunOutput
from llm_ensemble.libs.schemas.relevance_score import RelevanceScore

from tests.aggregate.mocks import (
    MockInputAdapter,
    MockOutputAdapter,
    MockAggregationStrategy,
)


@pytest.mark.integration
def test_aggregate_pipeline_with_real_majority_vote(
    infer_run_outputs_three_models: list[InferRunOutput],
    temp_run_dir: Path
):
    """Test end-to-end aggregation pipeline with real MajorityVoteAverage adapter.

    This demonstrates the testing benefits of Ports & Adapters architecture:
    - REAL domain logic (MajorityVoteAverage strategy)
    - MOCK infrastructure (file I/O)
    - Fast execution without external dependencies

    Tests application business logic:
    - Multiple InferRunOutputs are read and grouped by sample
    - Real majority vote logic is applied to each group
    - Aggregated votes have correct final labels
    - Summary statistics are calculated correctly
    """
    # Arrange: Real strategy + mock I/O
    output_adapter = MockOutputAdapter()
    app = AggregationApplication(
        reader=MockInputAdapter(infer_run_outputs_three_models),
        writer=output_adapter,
        aggregation_strategy=MajorityVoteAverage(strategy_name="majority_vote"),
        run_dir=temp_run_dir,
        run_name="integration-test-run",
    )

    # Act: Aggregate three model runs
    summary = app.run_aggregation(
        input_run_names=["run1", "run2", "run3"],
        official=False,
        notes="Integration test with real majority vote",
    )

    # Assert: Correct number of unique samples aggregated
    assert summary.unique_pair_count == 3
    assert summary.output_aggregated_count == 3

    # Assert: Total judgements from all runs (3 samples * 3 models = 9)
    assert summary.input_judgement_count == 9

    # Assert: Output adapter received the aggregate run
    assert output_adapter.written_aggregate_run is not None

    # Assert: Write summary tracks created entities
    assert summary.write_summary.aggregated_votes_created == 3
    assert summary.write_summary.aggregation_strategies_created == 1

    # Assert: No ties or invalid votes in this test data
    assert summary.tie_count == 0
    assert summary.no_valid_votes_count == 0


@pytest.mark.integration
def test_aggregate_pipeline_with_mock_strategy(
    infer_run_outputs_three_models: list[InferRunOutput],
    temp_run_dir: Path
):
    """Test aggregation pipeline with mock strategy to verify flow.

    Uses mock strategy to focus on testing pipeline orchestration
    rather than voting logic.
    """
    # Arrange: Mock everything
    input_adapter = MockInputAdapter(infer_run_outputs_three_models)
    output_adapter = MockOutputAdapter()
    mock_strategy = MockAggregationStrategy(strategy_name="mock_vote")

    app = AggregationApplication(
        reader=input_adapter,
        writer=output_adapter,
        aggregation_strategy=mock_strategy,
        run_dir=temp_run_dir,
        run_name="mock-strategy-test",
    )

    # Act
    summary = app.run_aggregation(
        input_run_names=["run1", "run2", "run3"],
        official=False,
        notes="Testing with mock strategy",
    )

    # Assert: Input adapter was called correctly
    assert input_adapter.read_called
    assert input_adapter.read_call_args == ["run1", "run2", "run3"]

    # Assert: Strategy was called for each unique sample
    assert len(mock_strategy.aggregate_calls) == 3

    # Assert: Each call had 3 judgements (one per model)
    for call in mock_strategy.aggregate_calls:
        assert len(call) == 3


@pytest.mark.integration
def test_aggregate_pipeline_official_run(
    infer_run_outputs_three_models: list[InferRunOutput],
    temp_run_dir: Path
):
    """Test aggregation pipeline marks official runs correctly.

    Verifies that the official flag is properly propagated through
    the pipeline and reflected in the run entity.
    """
    # Arrange
    app = AggregationApplication(
        reader=MockInputAdapter(infer_run_outputs_three_models),
        writer=MockOutputAdapter(),
        aggregation_strategy=MajorityVoteAverage(strategy_name="majority_vote"),
        run_dir=temp_run_dir,
        run_name="official-test-run",
    )

    # Act: Run as official
    summary = app.run_aggregation(
        input_run_names=["run1", "run2", "run3"],
        official=True,
        notes="Official run test",
    )

    # Assert: Run is marked as official
    from llm_ensemble.libs.runtime.run_info import RunType
    assert summary.run.run_type == RunType.OFFICIAL


@pytest.mark.integration
def test_aggregate_pipeline_timing(
    infer_run_outputs_three_models: list[InferRunOutput],
    temp_run_dir: Path
):
    """Test that aggregation pipeline records timing information.

    Verifies that start and end times are captured and that
    end time is after start time.
    """
    # Arrange
    app = AggregationApplication(
        reader=MockInputAdapter(infer_run_outputs_three_models),
        writer=MockOutputAdapter(),
        aggregation_strategy=MajorityVoteAverage(strategy_name="majority_vote"),
        run_dir=temp_run_dir,
        run_name="timing-test-run",
    )

    # Act
    summary = app.run_aggregation(
        input_run_names=["run1", "run2", "run3"],
        official=False,
        notes=None,
    )

    # Assert: Timing is recorded
    assert summary.start_time is not None
    assert summary.end_time is not None
    assert summary.end_time >= summary.start_time


@pytest.mark.integration
def test_aggregate_pipeline_run_metadata(
    infer_run_outputs_three_models: list[InferRunOutput],
    temp_run_dir: Path
):
    """Test that aggregation pipeline captures run metadata correctly.

    Verifies run name, notes, and strategy name are preserved.
    """
    # Arrange
    output_adapter = MockOutputAdapter()
    app = AggregationApplication(
        reader=MockInputAdapter(infer_run_outputs_three_models),
        writer=output_adapter,
        aggregation_strategy=MajorityVoteAverage(strategy_name="majority_vote"),
        run_dir=temp_run_dir,
        run_name="metadata-test-run",
    )

    # Act
    summary = app.run_aggregation(
        input_run_names=["run1", "run2", "run3"],
        official=False,
        notes="Test notes for metadata",
    )

    # Assert: Run metadata is correct
    assert summary.run.run_name == "metadata-test-run"
    assert summary.run.notes == "Test notes for metadata"

    # Assert: Config captures input run names
    assert summary.run.aggregate_run_config.input_run_names == ["run1", "run2", "run3"]

    # Assert: Strategy name is captured
    assert summary.run.aggregate_run_config.aggregation_strategy.name == "majority_vote"
