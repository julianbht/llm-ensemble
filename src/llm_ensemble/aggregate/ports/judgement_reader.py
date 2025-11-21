"""Port interface for judgement readers.

Defines the abstract contract for reading JudgedDataset from storage.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

from llm_ensemble.infer.schemas.judged_dataset import JudgedDataset


class JudgementReader(ABC):
    """Abstract base class for reading JudgedDataset from infer runs.

    Implementations can read from different formats (JSON, SQL, etc.)
    while providing a consistent interface.

    Readers accept run_name strings and load the complete JudgedDataset
    (fingerprint + judgements) for each run. All JudgedDatasets must have
    the same fingerprint to be aggregated together.
    """

    @abstractmethod
    def read(self, run_names: list[str]) -> list[JudgedDataset]:
        """Read JudgedDataset from one or more infer runs.

        Each run produces one JudgedDataset containing:
        - Fingerprint (SHA256 hash of sorted LLMCall IDs)
        - Judgements (LLM outputs for those calls)

        Validation: All JudgedDatasets must have the same fingerprint,
        ensuring they processed the same samples.

        Args:
            run_names: List of infer run identifiers (e.g., ["run1", "run2"])
                      Readers use PathManager or DB queries to resolve data

        Returns:
            List of JudgedDataset objects, one per run

        Raises:
            FileNotFoundError: If any run directory or expected files don't exist
            ValueError: If run names are invalid, data is malformed, or
                       fingerprints don't match across runs
        """
        pass
