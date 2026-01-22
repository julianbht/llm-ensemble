"""Port interface for judgement readers.

Defines the abstract contract for reading InferRunOutput from storage.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

from llm_ensemble.infer.domain.entities.infer_run_output import InferRunOutput


class ForInput(ABC):
    """Abstract base class for reading InferRunOutput from infer runs.

    Implementations can read from different formats (JSON, SQL, etc.)
    while providing a consistent interface.
    """

    @abstractmethod
    def read(self, run_names: list[str]) -> list[InferRunOutput]:
        """Read InferRunOutput from one or more infer runs.

        Args:
            run_names: List of infer run identifiers (e.g., ["run1", "run2"])

        Returns:
            List of InferRunOutput objects, one per run

        Raises:
            FileNotFoundError: If any run directory or expected files don't exist
            ValueError: If run names are invalid or data is malformed
        """
        pass
