"""Driving port for running data ingestion.

Driving Port (Primary/Driver Port) - Application API

This is the interface that the application OFFERS to driving adapters.
Driving adapters (CLI, Web API, Test harness, etc.) call this interface.

Defined BY the application, implemented BY the application (IngestApplication).
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
from pathlib import Path

from llm_ensemble.ingest.domain.entities.ingest_run_summary import IngestRunSummary


class ForRunningIngest(ABC):
    """Driving port for executing data ingestion pipeline.

    This is the application's public API that driving adapters use to
    trigger ingest runs. The application (IngestApplication) implements
    this interface, and driving adapters (CLI, Web API) call it.

    The application handles all backend concerns:
    - Logging configuration
    - Dataset normalization
    - Result persistence
    - Summary generation

    Infrastructure setup (run directories, run naming, tags) is handled
    by the composition root before the application is instantiated.
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

        Runs normalization and returns results.
        All logging appears in the configured output (terminal for CLI, CloudWatch for web, etc.).

        Workflow:
        1. Setup logging
        2. Read and normalize raw dataset via DatasetReader port
        3. Write normalized samples via DatasetWriter port
        4. Write summary and finalize outputs
        5. Return summary statistics

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
