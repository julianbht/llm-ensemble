"""Mock adapter implementations for testing the aggregate pipeline.

This module provides test doubles (mocks) for all driven ports in the aggregate CLI.
These mocks replace real infrastructure (file I/O, databases) with
in-memory implementations, enabling fast and deterministic testing of business logic.

Mock adapters are reusable across the aggregate test suite.
"""

from __future__ import annotations
from typing import Optional

from llm_ensemble.aggregate.application.ports.driven.for_input import ForInput
from llm_ensemble.aggregate.application.ports.driven.for_output import ForOutput
from llm_ensemble.aggregate.application.ports.driven.for_aggregating import ForAggregating
from llm_ensemble.aggregate.domain.entities.aggregate_run import AggregateRun
from llm_ensemble.aggregate.domain.entities.aggregated_vote import AggregatedVote
from llm_ensemble.aggregate.domain.entities.aggregation_strategy import AggregationStrategy
from llm_ensemble.aggregate.domain.entities.write_summary import WriteSummary
from llm_ensemble.infer.domain.entities.infer_run_output import InferRunOutput
from llm_ensemble.infer.domain.entities.llm_judgement import LLMJudgement


class MockInputAdapter(ForInput):
    """Mock input adapter that returns predefined InferRunOutputs.

    Replaces real file/database readers with in-memory test data.
    Allows testing without file I/O dependencies.
    """

    def __init__(self, mock_outputs: list[InferRunOutput]):
        """Initialize with test outputs.

        Args:
            mock_outputs: Predefined InferRunOutputs to return
        """
        self.mock_outputs = mock_outputs
        self.read_called = False
        self.read_call_args: Optional[list[str]] = None

    def read(self, run_names: list[str]) -> list[InferRunOutput]:
        """Return mock outputs and track call."""
        self.read_called = True
        self.read_call_args = run_names
        return self.mock_outputs


class MockOutputAdapter(ForOutput):
    """Mock output adapter that captures written data in memory.

    Replaces real file/database writers with in-memory collection.
    Allows verification of what was written without actual I/O.
    """

    def __init__(self):
        """Initialize empty collections for tracking writes."""
        self.written_aggregate_run: Optional[AggregateRun] = None
        self._write_summary = WriteSummary()

    @property
    def io_name(self) -> str:
        """Return adapter name."""
        return "mock"

    def write(self, aggregate_run: AggregateRun) -> WriteSummary:
        """Capture aggregate run in memory and return summary."""
        self.written_aggregate_run = aggregate_run

        # Track what would be written
        self._write_summary.add_aggregation_strategies(created=1)
        self._write_summary.add_configs(created=1)
        self._write_summary.add_aggregate_runs(created=1)

        if aggregate_run.aggregated_dataset:
            vote_count = len(aggregate_run.aggregated_dataset.aggregated_votes)
            self._write_summary.add_aggregated_datasets(created=1)
            self._write_summary.add_aggregated_votes(created=vote_count)

        return self._write_summary


class MockAggregationStrategy(ForAggregating):
    """Mock aggregation strategy that returns predefined votes.

    Replaces real aggregation logic with deterministic results.
    Useful for testing pipeline flow without complex voting logic.
    """

    def __init__(self, strategy_name: str = "mock_strategy"):
        """Initialize with strategy name.

        Args:
            strategy_name: Name for the mock strategy
        """
        self._strategy_name = strategy_name
        self.aggregate_calls: list[list[LLMJudgement]] = []

    def aggregate(self, judgements: list[LLMJudgement]) -> AggregatedVote:
        """Create aggregated vote from judgements and track call."""
        from llm_ensemble.aggregate.domain.aggregated_vote_builder import build_aggregated_vote
        from llm_ensemble.libs.schemas.relevance_score import RelevanceScore

        self.aggregate_calls.append(judgements)

        # Simple mock: use first valid label or None
        final_label = None
        for j in judgements:
            if j.llm_score and j.llm_score.label is not None:
                final_label = j.llm_score.label
                break

        return build_aggregated_vote(
            llm_judgements=judgements,
            final_label=final_label,
            final_confidence=1.0 if final_label else 0.0,
            final_reasoning="Mock aggregation result",
        )

    def get_strategy(self) -> AggregationStrategy:
        """Return strategy metadata."""
        return AggregationStrategy(name=self._strategy_name)
