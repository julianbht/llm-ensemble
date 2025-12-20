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

from llm_ensemble.infer.domain.entities.infer_run_config import InferRunConfig
from llm_ensemble.infer.startup.adapter_config import ExecutionParams
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
        run_config: InferRunConfig,
        execution_params: ExecutionParams,
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
            run_config: Configuration bundle (model, provider, adapters, execution context)
            execution_params: Execution parameters (run name, official flag, notes, tag, etc.)

        Returns:
            InferRunSummary with statistics, timing, and warnings

        Raises:
            Exception: If any step in the pipeline fails
        """
        pass
