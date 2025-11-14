"""Port interface for reading judging examples.

Defines the abstract contract for reading examples from various sources
(JSON files, Parquet, databases, etc.). This allows the orchestrator
to work with any input format without coupling to a specific implementation.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from llm_ensemble.ingest.schemas import JudgingSample


class ExampleReader(ABC):
    """Abstract base class for reading judging examples.

    Implementations can read from different sources (JSON, Parquet, etc.)
    while providing a consistent interface to the orchestrator.

    Example:
        >>> class JsonExampleReader(ExampleReader):
        ...     def read(self, input_path, limit=None):
        ...         examples = []
        ...         with open(input_path) as f:
        ...             for line in f:
        ...                 examples.append(JudgingSample(**json.loads(line)))
        ...         return examples[:limit] if limit else examples
    """

    @abstractmethod
    def read(
        self,
        input_path: Path,
        limit: Optional[int] = None,
    ) -> list[JudgingSample]:
        """Read examples from source.

        Args:
            input_path: Path to input file/resource
            limit: Optional maximum number of examples to read

        Returns:
            List of JudgingSample objects

        Raises:
            FileNotFoundError: If input_path doesn't exist (file-based readers)
            ValueError: If input is invalid (e.g., invalid run name for DB readers, malformed data)
        """
        pass
