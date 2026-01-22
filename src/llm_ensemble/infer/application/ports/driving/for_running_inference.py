"""Driving port for running LLM inference.

Driving Port (Primary/Driver Port) - Application API

This is the interface that the application OFFERS to driving adapters.
Driving adapters (CLI, Web API, Test harness, etc.) call this interface.

Defined BY the application, implemented BY the application (InferenceApplication).
Called BY driving adapters.

In hexagonal architecture, this represents the hexagon's edge facing outward
toward the driving adapters.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

from llm_ensemble.infer.domain.entities.infer_run_summary import InferRunSummary


class ForRunningInference(ABC):
    """
    Driving port for executing LLM inference pipeline.
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
