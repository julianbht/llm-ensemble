"""Port interface for reading judging examples.

Defines the abstract contract for reading examples from various sources
(NDJSON files, Parquet, databases, etc.). This allows the orchestrator
to work with any input format without coupling to a specific implementation.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from llm_ensemble.ingest.schemas import JudgingExample


class ExampleReader(ABC):
    """Abstract base class for reading judging examples.

    Implementations can read from different sources (NDJSON, Parquet, etc.)
    while providing a consistent interface to the orchestrator.

    Example:
        >>> class NdjsonExampleReader(ExampleReader):
        ...     def read(self, input_path, limit=None):
        ...         examples = []
        ...         with open(input_path) as f:
        ...             for line in f:
        ...                 examples.append(JudgingExample(**json.loads(line)))
        ...         return examples[:limit] if limit else examples
    """

    @abstractmethod
    def read(
        self,
        input_path: Path,
        limit: Optional[int] = None,
    ) -> list[JudgingExample]:
        """Read examples from source.

        Args:
            input_path: Path to input file/resource
            limit: Optional maximum number of examples to read

        Returns:
            List of JudgingExample objects

        Raises:
            FileNotFoundError: If input_path doesn't exist
            ValueError: If file format is invalid
        """
        pass
