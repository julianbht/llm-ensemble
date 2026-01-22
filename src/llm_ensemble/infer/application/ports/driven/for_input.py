"""Port interface for reading NormalizedDataset.

Defines the abstract contract for reading NormalizedDataset from various sources
(JSON files, Parquet, databases, etc.).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

from llm_ensemble.ingest.domain.entities.normalized_dataset import NormalizedDataset


class ForInput(ABC):
    """Abstract base class for reading NormalizedDataset from ingest runs.

    Implementations can read from different sources (JSON, Parquet, SQL, etc.)
    while providing a consistent interface to the application.
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
            limit: Optional maximum number of samples to include in the dataset

        Returns:
            NormalizedDataset with samples, fingerprint, and metadata

        Raises:
            FileNotFoundError: If run directory or expected files don't exist
            ValueError: If run name is invalid or data is malformed
        """
        pass
