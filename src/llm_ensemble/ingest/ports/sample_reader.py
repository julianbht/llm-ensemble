"""Port interface for reading judging samples.

Defines the abstract contract for reading raw IR datasets and converting them
to RawSample DTOs (query + document + gold score, without manifest).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from llm_ensemble.ingest.schemas import Query, Document, RelevanceScore


@dataclass(frozen=True)
class RawJudgingSample:
    """DTO for transferring data across the SampleReader port boundary.

    This is NOT a persisted schema - it's purely for internal data transfer
    between adapters and the domain service. The domain service is responsible
    for attaching the manifest to create the full JudgingSample.

    DO NOT export this from schemas/ - it lives only at the port boundary.
    """
    query: Query
    document: Document
    gold_score: RelevanceScore


class SampleReader(ABC):
    """Abstract base class for reading judging samples from raw datasets.

    Implementations read dataset-specific formats (TSV + JSONL, Parquet, etc.)
    and convert them to RawJudgingSample DTOs (without manifest).

    The domain service is responsible for attaching the manifest to create
    full JudgingSample objects.

    Example:
        >>> class LlmJudgeSampleReader(SampleReader):
        ...     def read(self, input_path, limit=None):
        ...         # Read queries, documents, qrels
        ...         # Convert to RawJudgingSample DTOs
        ...         return samples[:limit] if limit else samples
    """

    @abstractmethod
    def read(
        self,
        input_path: Path,
        dataset_name: str,
        limit: Optional[int] = None,
    ) -> list[RawJudgingSample]:
        """Read raw dataset and return RawJudgingSample DTOs (without manifest).

        Args:
            input_path: Path to input dataset (file or directory)
            dataset_name: Dataset identifier for computing deterministic UUIDs
            limit: Optional maximum number of samples to read

        Returns:
            List of RawJudgingSample DTOs (with IDs computed from dataset_name)

        Raises:
            FileNotFoundError: If input_path doesn't exist
            ValueError: If dataset format is invalid
        """
        pass
