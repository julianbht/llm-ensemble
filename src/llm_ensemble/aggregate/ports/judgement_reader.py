"""Port interface for judgement readers.

Defines the abstract contract for reading JudgedDataset from storage.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

from llm_ensemble.infer.schemas.entities.judged_dataset import JudgedDataset


class JudgementReader(ABC):
    """Abstract base class for reading JudgedDataset from infer runs.

    Implementations can read from different formats (JSON, SQL, etc.)
    while providing a consistent interface.

    Readers are "dumb" data loaders - they simply fetch data without
    performing validation. Validation logic belongs in the service layer.
    """

    @abstractmethod
    def read(self, run_names: list[str]) -> list[JudgedDataset]:
        """Read JudgedDataset from one or more infer runs.

        Each run produces one JudgedDataset containing:
        - Fingerprint (SHA256 hash of sorted LLMCall IDs)
        - Judgements (LLM outputs for those calls)

        Readers do not perform validation - the service layer handles:
        - Checking for NULL fingerprints
        - Validating fingerprints match across runs
        - Ensuring runs completed successfully

        Args:
            run_names: List of infer run identifiers (e.g., ["run1", "run2"])
                      Readers use PathManager or DB queries to resolve data

        Returns:
            List of JudgedDataset objects, one per run

        Raises:
            FileNotFoundError: If any run directory or expected files don't exist
            ValueError: If run names are invalid or data is malformed
        """
        pass
