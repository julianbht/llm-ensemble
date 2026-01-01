"""Dummy output adapter for testing infrastructure.

Driven Adapter - I/O (Hexagonal Architecture)

Implements ForOutput port with a placeholder writer.
This adapter demonstrates the output adapter pattern and will be
replaced with real implementations that write HTML/JSON reports.
"""

from __future__ import annotations
from typing import Any

from llm_ensemble.evaluate.application.ports.driven.for_output import ForOutput


class DummyWriter(ForOutput):
    """Dummy output adapter that prints to console.

    Implements ForOutput port for testing infrastructure.
    Replace with real implementations (HTML, JSON, etc.).
    """

    def __init__(self, io_name: str):
        """Initialize dummy writer.

        Args:
            io_name: I/O configuration name
        """
        self.io_name = io_name

    def write(self, metric_results: list[Any], run_metadata: Any) -> None:
        """Write placeholder report (prints to console).

        Args:
            metric_results: List of metric computation results
            run_metadata: Metadata about the evaluation run

        Raises:
            IOError: If writing fails (not implemented)
        """
        print("=" * 60)
        print("EVALUATION REPORT (Dummy Output)")
        print("=" * 60)
        print(f"\nMetadata: {run_metadata}")
        print(f"\nMetric Results:")
        for result in metric_results:
            print(f"  - {result}")
        print("=" * 60)
