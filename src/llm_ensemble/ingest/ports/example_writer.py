"""Abstract port for writing JudgingExample records.

Defines the contract for serializing and writing JudgingExample records
to persistent storage (files, databases, etc.).
"""

from __future__ import annotations
from abc import ABC, abstractmethod

from llm_ensemble.ingest.schemas import JudgingExample


class ExampleWriter(ABC):
    """Abstract base class for writing JudgingExample records.

    Example writers are responsible for serializing JudgingExample records
    and persisting them to storage. Different implementations can write to
    different formats (NDJSON, Parquet, etc.) or storage systems (local files,
    cloud storage, databases).

    Example:
        >>> writer = NdjsonExampleWriter(Path("output.ndjson"))
        >>> for example in examples:
        ...     writer.write(example)
        >>> writer.close()
    """

    @abstractmethod
    def write(self, example: JudgingExample) -> None:
        """Write a single JudgingExample record.

        Args:
            example: The JudgingExample to write

        Raises:
            IOError: If writing fails
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the writer and flush any buffered data.

        Should be called after all records have been written to ensure
        data is properly persisted.

        Raises:
            IOError: If flushing/closing fails
        """
        pass
