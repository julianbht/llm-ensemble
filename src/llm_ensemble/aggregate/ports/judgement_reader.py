"""Port interface for judgement readers.

Defines the abstract contract for reading InferredDataset from storage.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

from llm_ensemble.infer.schemas.inferred_dataset import InferredDataset


class JudgementReader(ABC):
    """Abstract base class for reading InferredDataset from infer runs.

    Implementations can read from different formats (JSON, SQL, etc.)
    while providing a consistent interface.

    Readers accept run_name strings and load the complete InferredDataset
    (fingerprint + judgements) for each run.
    """

    @abstractmethod
    def read(self, run_names: list[str]) -> list[InferredDataset]:
        """Read InferredDataset from one or more infer runs.

        Each run produces one InferredDataset containing:
        - Fingerprint (identifies which samples were processed)
        - Judgements (LLM outputs for those samples)

        Args:
            run_names: List of infer run identifiers (e.g., ["run1", "run2"])
                      Readers use PathManager or DB queries to resolve data

        Returns:
            List of InferredDataset objects, one per run

        Raises:
            FileNotFoundError: If any run directory or expected files don't exist
            ValueError: If run names are invalid or data is malformed
        """
        pass
