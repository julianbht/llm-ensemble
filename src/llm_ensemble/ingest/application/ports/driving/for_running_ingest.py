"""Driving port for running data ingestion.

Driving Port (Primary/Driver Port) - Application API

This is the interface that the application OFFERS to driving adapters.
Driving adapters (CLI, Web API, Test harness, etc.) call this interface.

Defined BY the application, implemented BY the application (IngestApplication).
Called BY driving adapters.

In hexagonal architecture, this represents the hexagon's edge facing outward
toward the driving adapters.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
from pathlib import Path

from llm_ensemble.ingest.domain.entities.ingest_run_summary import IngestRunSummary


class ForRunningIngest(ABC):
    """
    Driving port for executing data ingestion pipeline.
    """

    @abstractmethod
    def run_ingest(
        self,
        input_path: Path,
        limit: Optional[int],
        official: bool,
        notes: Optional[str],
    ) -> IngestRunSummary:
        """Execute the ingestion pipeline.

        Args:
            input_path: Path to input directory containing raw dataset files
            limit: Process at most N samples
            official: Mark as official run
            notes: Notes about this run (experiment purpose, hypothesis, etc.)

        Returns:
            IngestRunSummary with statistics and timing

        Raises:
            FileNotFoundError: If input path doesn't exist
            ValueError: If adapter is not recognized or dataset files are malformed
            Exception: If any step in the pipeline fails
        """
        pass
