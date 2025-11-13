"""Port interface for reading complete normalized datasets.

Defines the abstract contract for reading raw IR datasets and converting them
to NormalizedDataset objects containing both metadata and samples.

The reader is responsible for:
- Extracting dataset metadata from data
- Creating all domain objects (Dataset, Query, Document, JudgingSample)
- Computing deterministic UUIDs
- Returning a complete NormalizedDataset
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from llm_ensemble.ingest.schemas import NormalizedDataset


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

    Example:
        >>> class LlmJudgeDatasetReader(DatasetReader):
        ...     def read(self, input_path, limit=None):
        ...         # Extract dataset name from files
        ...         dataset = Dataset.create("llmjudge", "LLM Judge Challenge 2024")
        ...         # Read and normalize queries, documents, qrels
        ...         samples = [JudgingSample.create(...), ...]
        ...         return NormalizedDataset(dataset, samples)
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
            NormalizedDataset containing metadata and complete samples

        Raises:
            FileNotFoundError: If input_path doesn't exist
            ValueError: If dataset format is invalid
        """
        pass
