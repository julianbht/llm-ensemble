"""Port interface for reading complete normalized datasets.

Defines the abstract contract for reading raw IR datasets and converting them
to NormalizedDataset objects containing both metadata and samples.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

from llm_ensemble.ingest.domain.entities.normalized_dataset import NormalizedDataset


class DatasetReader(ABC):
    """Abstract base class for reading and normalizing IR datasets.

    Implementations read dataset-specific formats (TSV + JSONL, Parquet, etc.)
    and return a complete NormalizedDataset with metadata and samples.

    The reader extracts dataset information from the data itself (not from config),
    ensuring dataset metadata travels with the data.

    The reader handles complete normalization:
    - Dataset metadata extraction
    - Creating Query and Document entities with UUIDs
    - Creating complete JudgingSample objects
    - Packaging everything as NormalizedDataset
    
    Note: DatasetReader is for raw dataset ingestion, not reading from runs.
    It still accepts paths since it reads from external data sources.
    """

    @abstractmethod
    def read(
        self,
        input_path: str,
        limit: Optional[int] = None,
    ) -> NormalizedDataset:
        """Read and normalize raw dataset.

        Args:
            input_path: Path to input dataset (file or directory, as string)
            limit: Optional maximum number of samples to read

        Returns:
            NormalizedDataset containing metadata and complete samples

        Raises:
            FileNotFoundError: If input_path doesn't exist
            ValueError: If dataset format is invalid
        """
        pass
