"""Port interface for reading complete normalized datasets.

Defines the abstract contract for reading raw IR datasets and converting them
to NormalizedDataset objects containing both metadata and samples.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from llm_ensemble.ingest.domain.entities.normalized_dataset import NormalizedDataset


class ForInput(ABC):
    """Abstract base class for reading and normalizing IR datasets.

    The reader handles complete normalization:
    - Dataset metadata extraction
    - Creating Query and Document entities with UUIDs
    - Creating complete JudgingSample objects
    - Packaging everything as NormalizedDataset
    """

    @abstractmethod
    def read(
        self,
        input_path: Path,
        limit: Optional[int] = None,
    ) -> NormalizedDataset:
        """Read and normalize raw dataset.

        Args:
            input_path: Path to input dataset (file or directory)
            limit: Optional maximum number of samples to read

        Returns:
            NormalizedDataset containing metadata and samples

        Raises:
            FileNotFoundError: If input_path doesn't exist
            ValueError: If dataset format is invalid
        """
        pass
