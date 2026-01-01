"""Dummy input adapter for testing infrastructure.

Driven Adapter - I/O (Hexagonal Architecture)

Implements ForInput port with a placeholder reader.
This adapter demonstrates the input adapter pattern and will be
replaced with real implementations that read from infer/aggregate runs.
"""

from __future__ import annotations
from typing import Any

from llm_ensemble.evaluate.application.ports.driven.for_input import ForInput


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

    def read(self, input_run_name: str) -> Any:
        """Read placeholder evaluation data.

        Args:
            input_run_name: Run name to read from (ignored for dummy)

        Returns:
            Placeholder data structure

        Raises:
            FileNotFoundError: If input run not found (not implemented)
        """
        # Return placeholder data for testing
        return {
            "ground_truth": [1, 2, 3, 1, 2],
            "predictions": [1, 2, 2, 1, 3],
            "metadata": {
                "input_run_name": input_run_name,
                "sample_count": 5,
            },
        }
