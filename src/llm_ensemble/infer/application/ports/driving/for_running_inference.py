"""Driving port for running LLM inference.

Driving Port (Primary/Driver Port) - Application API

This is the interface that the application OFFERS to driving adapters.
Driving adapters (CLI, Web API, Test harness, etc.) call this interface.

Defined BY the application, implemented BY the application (InferenceUseCase).
Called BY driving adapters (CLIDriver, WebAPIDriver, etc.).

In hexagonal architecture, this represents the hexagon's edge facing outward
toward the driving adapters.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

from llm_ensemble.infer.domain.entities.infer_run_info import InferRunInfo
from llm_ensemble.infer.domain.entities.infer_run_config import InferRunConfig
from llm_ensemble.infer.schemas.infer_run_summary import InferRunSummary


class ForRunningInference(ABC):
    """Driving port for executing LLM inference pipeline.

    This is the application's public API that driving adapters use to
    trigger inference runs. The application (InferenceUseCase) implements
    this interface, and driving adapters (CLI, Web API) call it.

    Comparable to ForParkingCars or ForCheckingCars in BlueZone example.
    """

    @abstractmethod
    def execute(
        self,
        run_info: InferRunInfo,
        run_config: InferRunConfig,
    ) -> InferRunSummary:
        """Execute the inference pipeline.

        Coordinates the full inference workflow:
        1. Read dataset samples via InputPort
        2. For each sample: build prompt, call LLM, parse response, write result
        3. Calculate and return summary statistics

        Args:
            run_info: Run metadata (git SHA, timestamps, run type, notes)
            run_config: Configuration bundle (model, provider, adapters, execution context)

        Returns:
            InferRunSummary with statistics, timing, and warnings

        Raises:
            Exception: If any step in the pipeline fails
        """
        pass
