"""Port interface for reading judging samples.

Defines the abstract contract for reading raw IR datasets and converting them
to JudgingSample objects.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from llm_ensemble.ingest.schemas import JudgingSample


class SampleReader(ABC):
    """Abstract base class for reading judging samples from raw datasets.

    Implementations read dataset-specific formats (TSV + JSONL, Parquet, etc.)
    and convert them to normalized JudgingSample objects.

    Example:
        >>> class LlmJudgeSampleReader(SampleReader):
        ...     def read(self, input_path, limit=None):
        ...         # Read queries, documents, qrels
        ...         # Convert to JudgingSample objects
        ...         return samples[:limit] if limit else samples
    """

    @abstractmethod
    def read(
        self,
        input_path: Path,
        limit: Optional[int] = None,
    ) -> list[JudgingSample]:
        """Read raw dataset and return normalized JudgingSample objects.

        Args:
            input_path: Path to input dataset (file or directory)
            limit: Optional maximum number of samples to read

        Returns:
            List of JudgingSample objects

        Raises:
            FileNotFoundError: If input_path doesn't exist
            ValueError: If dataset format is invalid
        """
        pass
