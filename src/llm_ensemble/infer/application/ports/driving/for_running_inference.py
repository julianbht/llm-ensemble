"""Driving port for running LLM inference.

Driving Port (Primary/Driver Port) - Application API

This is the interface that the application OFFERS to driving adapters.
Driving adapters (CLI, Web API, Test harness, etc.) call this interface.

Defined BY the application, implemented BY the application (InferenceUseCase).
Called BY driving adapters (CLI, Web API, etc.).

In hexagonal architecture, this represents the hexagon's edge facing outward
toward the driving adapters.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

from llm_ensemble.infer.schemas.infer_run_summary import InferRunSummary


class ForRunningInference(ABC):
    """Driving port for executing LLM inference pipeline.

    This is the application's public API that driving adapters use to
    trigger inference runs. The application (InferenceUseCase) implements
    this interface, and driving adapters (CLI, Web API) call it.

    The application handles all backend concerns:
    - Infrastructure setup (run directories, logging)
    - Inference execution
    - Result persistence
    - Summary generation
    """

    @abstractmethod
    def execute(
        self,
        input_run_name: str,
        start_idx: Optional[int],
        end_idx: Optional[int],
        run_name: Optional[str],
        official: bool,
        notes: Optional[str],
        tag: Optional[str],
    ) -> InferRunSummary:
        """Execute the inference pipeline with full backend infrastructure.

        Sets up infrastructure, runs inference, and returns results.
        All logging appears in the configured output (terminal for CLI, CloudWatch for web, etc.).

        Workflow:
        1. Setup infrastructure (run directories, logging, git metadata)
        2. Read dataset samples via InputPort
        3. For each sample: build prompt, call LLM, parse response, write result
        4. Write summary and finalize outputs
        5. Return summary statistics

        Args:
            input_run_name: Ingest run identifier to read samples from
            start_idx: Start index into NormalizedDataset (None = from beginning)
            end_idx: End index into NormalizedDataset (None = until end)
            run_name: Custom run name (auto-generates if not provided)
            official: Mark as official run
            notes: Notes about this run (experiment purpose, hypothesis, etc.)
            tag: Tag name for easy reference by downstream CLIs

        Returns:
            InferRunSummary with statistics, timing, and warnings

        Raises:
            Exception: If any step in the pipeline fails
        """
        pass
