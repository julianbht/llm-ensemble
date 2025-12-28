"""Port interface for reading NormalizedDataset.

Defines the abstract contract for reading NormalizedDataset from various sources
(JSON files, Parquet, databases, etc.). This allows the orchestrator
to work with any input format without coupling to a specific implementation.

By reading the full NormalizedDataset entity (not just individual samples),
the INFER pipeline can:
- Know the complete set of samples to judge
- Enable resumability by comparing what's been judged vs what remains
- Track clear provenance of input data
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

from llm_ensemble.ingest.schemas.normalized_dataset import NormalizedDataset


class ForInput(ABC):
    """Abstract base class for reading NormalizedDataset from ingest runs.

    Implementations can read from different sources (JSON, Parquet, SQL, etc.)
    while providing a consistent interface to the orchestrator.

    Readers accept run_name strings and internally resolve to file paths or
    database queries, enabling clean separation of concerns.
    """

    @abstractmethod
    def read(
        self,
        run_name: str,
        limit: Optional[int] = None,
    ) -> NormalizedDataset:
        """Read NormalizedDataset from ingest run.

        Args:
            run_name: Run identifier (e.g., "my_ingest_run" or "20250128_143022")
                     Readers use PathManager to resolve to appropriate paths/queries
            limit: Optional maximum number of samples to include in the dataset
                  (useful for testing or partial runs)

        Returns:
            NormalizedDataset with samples, fingerprint, and metadata

        Raises:
            FileNotFoundError: If run directory or expected files don't exist
            ValueError: If run name is invalid or data is malformed
        """
        pass
