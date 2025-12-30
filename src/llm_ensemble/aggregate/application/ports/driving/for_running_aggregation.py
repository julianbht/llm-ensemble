"""Driving port for running aggregation.

Driving Port (Primary/Driver Port) - Application API

This is the interface that the application OFFERS to driving adapters.
Driving adapters (CLI, Web API, Test harness, etc.) call this interface.

Defined BY the application, implemented BY the application (AggregationApplication).
Called BY driving adapters (CLI, Web API, etc.).

In hexagonal architecture, this represents the hexagon's edge facing outward
toward the driving adapters.

Note: Run directory and run name are provided at construction time via the
composition root, not through this interface. This keeps infrastructure
concerns separate from business logic.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

from llm_ensemble.aggregate.domain.entities.aggregate_run_summary import AggregateRunSummary


class ForRunningAggregation(ABC):
    """Driving port for executing aggregation pipeline.

    This is the application's public API that driving adapters use to
    trigger aggregation runs. The application (AggregationApplication) implements
    this interface, and driving adapters (CLI, Web API) call it.

    The application handles all backend concerns:
    - Logging configuration
    - Aggregation execution
    - Result persistence
    - Summary generation

    Infrastructure setup (run directories, run naming, tags) is handled
    by the composition root before the application is instantiated.
    """

    @abstractmethod
    def run_aggregation(
        self,
        input_run_names: list[str],
        official: bool,
        notes: Optional[str],
    ) -> AggregateRunSummary:
        """Execute the aggregation pipeline.

        Runs aggregation and returns results.
        All logging appears in the configured output (terminal for CLI, CloudWatch for web, etc.).

        Workflow:
        1. Setup logging
        2. Read InferRunOutputs via JudgementReader port
        3. Validate sample_fingerprints match
        4. Group judgements by dataset_sample_id
        5. Apply aggregation strategy to each group
        6. Create AggregatedDataset
        7. Build AggregateRun entity
        8. Write AggregateRun via writer port (batch persistence)
        9. Write summary and finalize outputs
        10. Return summary statistics

        Args:
            input_run_names: List of infer run identifiers to read judgements from
            official: Mark as official run
            notes: Notes about this run (experiment purpose, hypothesis, etc.)

        Returns:
            AggregateRunSummary with statistics, timing, and warnings

        Raises:
            ValueError: If validation fails or strategy not found
            FileNotFoundError: If any run directory doesn't exist
            Exception: If any step in the pipeline fails
        """
        pass
