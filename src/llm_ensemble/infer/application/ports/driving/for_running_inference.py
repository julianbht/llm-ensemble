"""Driving port for running LLM inference.

Driving Port (Primary/Driver Port) - Application API

This is the interface that the application OFFERS to driving adapters.
Driving adapters (CLI, Web API, Test harness, etc.) call this interface.

Defined BY the application, implemented BY the application (InferenceUseCase).
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

from llm_ensemble.infer.domain.entities.infer_run_summary import InferRunSummary


class ForRunningInference(ABC):
    """Driving port for executing LLM inference pipeline.

    This is the application's public API that driving adapters use to
    trigger inference runs. The application (InferenceUseCase) implements
    this interface, and driving adapters (CLI, Web API) call it.

    The application handles all backend concerns:
    - Logging configuration
    - Inference execution
    - Result persistence
    - Summary generation

    Infrastructure setup (run directories, run naming, tags) is handled
    by the composition root before the application is instantiated.
    """

    @abstractmethod
    def run_inference(
        self,
        input_run_name: str,
        start_idx: Optional[int],
        end_idx: Optional[int],
        official: bool,
        notes: Optional[str],
    ) -> InferRunSummary:
        """Execute the inference pipeline.

        Runs inference and returns results.
        All logging appears in the configured output (terminal for CLI, CloudWatch for web, etc.).

        Workflow:
        1. Setup logging
        2. Read dataset samples via InputPort
        3. For each sample: build prompt, call LLM, parse response, write result
        4. Write summary and finalize outputs
        5. Return summary statistics

        Args:
            input_run_name: Ingest run identifier to read samples from
            start_idx: Start index into NormalizedDataset (None = from beginning)
            end_idx: End index into NormalizedDataset (None = until end)
            official: Mark as official run
            notes: Notes about this run (experiment purpose, hypothesis, etc.)

        Returns:
            InferRunSummary with statistics, timing, and warnings

        Raises:
            Exception: If any step in the pipeline fails
        """
        pass
