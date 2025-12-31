"""Factory for creating AggregateRun entities.

Domain Layer - Factory Pattern

Creates AggregateRun aggregate root from domain entities and primitive values,
assembling the complete record for manifest persistence.

This factory belongs in the domain layer because it only depends on
domain entities and performs pure assembly logic. The application layer
is responsible for providing these values.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from llm_ensemble.aggregate.domain.entities.aggregate_run import AggregateRun
from llm_ensemble.aggregate.domain.entities.aggregate_run_config import AggregateRunConfig
from llm_ensemble.aggregate.domain.entities.aggregated_vote import AggregatedVote
from llm_ensemble.aggregate.domain.entities.aggregation_strategy import AggregationStrategy
from llm_ensemble.aggregate.domain.aggregated_dataset_builder import build_aggregated_dataset
from llm_ensemble.libs.runtime.run_info import RunType


class AggregateRunFactory:
    """Factory for creating AggregateRun aggregate root from domain entities.

    Domain layer factory - pure assembly logic with no adapter dependencies.
    """

    @staticmethod
    def create(
        aggregation_strategy_name: str,
        io_config_name: str,
        input_run_names: list[str],
        run_name: str,
        run_type: RunType,
        aggregated_votes: list[AggregatedVote],
        start_time: datetime,
        end_time: datetime,
        notes: Optional[str],
    ) -> AggregateRun:
        """Create AggregateRun aggregate root from domain entities and primitive values.

        Args:
            aggregation_strategy_name: Name of the aggregation strategy used
            io_config_name: Name of the I/O configuration used
            input_run_names: List of infer run identifiers to read judgements from
            run_name: Run identifier
            run_type: Type of run (OFFICIAL or TEST)
            aggregated_votes: List of aggregated votes produced by this run
            start_time: When the run started
            end_time: When the run completed
            notes: Notes about this run (experiment purpose, hypothesis, etc.)

        Returns:
            Assembled AggregateRun aggregate root
        """
        # Build AggregationStrategy entity
        aggregation_strategy = AggregationStrategy(name=aggregation_strategy_name)

        # Build AggregateRunConfig entity
        run_config = AggregateRunConfig(
            aggregation_strategy=aggregation_strategy,
            io_config_name=io_config_name,
            input_run_names=input_run_names,
        )

        # Build AggregatedDataset from votes (computes fingerprint and UUID)
        aggregated_dataset = build_aggregated_dataset(aggregated_votes)

        # Assemble complete aggregate root
        return AggregateRun(
            run_name=run_name,
            run_type=run_type,
            aggregate_run_config=run_config,
            aggregated_dataset=aggregated_dataset,
            start_time=start_time,
            end_time=end_time,
            notes=notes,
        )
