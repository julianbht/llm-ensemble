"""Driving port for running aggregation.

Driving Port (Primary/Driver Port) - Application API

This is the interface that the application OFFERS to driving adapters.
Driving adapters (CLI, Web API, Test harness, etc.) call this interface.

Defined BY the application, implemented BY the application (AggregationApplication).
Called BY driving adapters.

In hexagonal architecture, this represents the hexagon's edge facing outward
toward the driving adapters.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

from llm_ensemble.aggregate.domain.entities.aggregate_run_summary import AggregateRunSummary


class ForRunningAggregation(ABC):
    """
    Driving port for executing aggregation pipeline.
    """

    @abstractmethod
    def run_aggregation(
        self,
        input_run_names: list[str],
        official: bool,
        notes: Optional[str],
    ) -> AggregateRunSummary:
        """Execute the aggregation pipeline.

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
