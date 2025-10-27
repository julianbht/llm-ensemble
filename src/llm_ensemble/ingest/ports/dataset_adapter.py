"""Abstract port for dataset adapters.

Defines the contract for reading raw IR datasets and converting them
into normalized JudgingExample records.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

from llm_ensemble.ingest.schemas import JudgingExample


class DatasetAdapter(ABC):
    """Abstract base class for dataset adapters.

    Dataset adapters are responsible for reading raw IR dataset files
    (queries, documents, relevance judgements) and yielding normalized
    JudgingExample records.

    Each dataset has its own format and file structure, so concrete adapters
    implement the specific parsing logic for each dataset.

    Example:
        >>> adapter = LlmJudgeAdapter()
        >>> examples = adapter.read(Path("/data/llm-judge-2024"))
        >>> for example in examples:
        ...     process(example)
    """

    @abstractmethod
    def read(self, data_dir: Path) -> Iterator[JudgingExample]:
        """Read raw dataset files and yield normalized JudgingExamples.

        Args:
            data_dir: Base directory containing the dataset files

        Yields:
            JudgingExample: Normalized judging examples

        Raises:
            FileNotFoundError: If required dataset files are missing
            ValueError: If dataset files are malformed
        """
        pass
