"""Dummy input adapter for testing infrastructure.

Driven Adapter - I/O (Hexagonal Architecture)

Implements ForInput port with a placeholder reader.
This adapter demonstrates the input adapter pattern and will be
replaced with real implementations that read from infer/aggregate runs.
"""

from __future__ import annotations

from llm_ensemble.evaluate.application.ports.driven.for_input import ForInput
from llm_ensemble.evaluate.domain.entities.evaluation_data import EvaluationData
from llm_ensemble.evaluate.domain.evaluation_data_builder import build_evaluation_data
from llm_ensemble.libs.schemas.relevance_score import RelevanceScore


class DummyReader(ForInput):
    """Dummy input adapter that returns placeholder data.

    Implements ForInput port for testing infrastructure.
    Replace with real implementations that read from run directories.
    """

    def __init__(self, io_name: str):
        """Initialize dummy reader.

        Args:
            io_name: I/O configuration name
        """
        self.io_name = io_name

    def read(self, input_run_name: str) -> EvaluationData:
        """Read placeholder evaluation data.

        Args:
            input_run_name: Run name to read from (ignored for dummy)

        Returns:
            EvaluationData entity with placeholder data

        Raises:
            FileNotFoundError: If input run not found (not implemented)
        """
        # Create placeholder data using domain builder
        return build_evaluation_data(
            ground_truth=[
                RelevanceScore.RELEVANT,
                RelevanceScore.HIGHLY_RELEVANT,
                RelevanceScore.PERFECTLY_RELEVANT,
                RelevanceScore.RELEVANT,
                RelevanceScore.HIGHLY_RELEVANT,
            ],
            predictions=[
                RelevanceScore.RELEVANT,
                RelevanceScore.HIGHLY_RELEVANT,
                RelevanceScore.HIGHLY_RELEVANT,
                RelevanceScore.RELEVANT,
                RelevanceScore.PERFECTLY_RELEVANT,
            ],
            run_name=input_run_name,
            run_type="dummy",
        )
