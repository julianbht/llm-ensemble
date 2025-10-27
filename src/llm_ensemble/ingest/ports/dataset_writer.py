"""Port interface for writing complete NormalizedDataset.

Defines the abstract contract for writing a complete NormalizedDataset
(samples + manifest) to persistent storage.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path

from llm_ensemble.ingest.schemas import NormalizedDataset


class DatasetWriter(ABC):
    """Abstract base class for writing NormalizedDataset.

    Writes the entire NormalizedDataset (judging samples + manifest) as a
    single artifact. This ensures the manifest is always bundled with the data.

    Example:
        >>> writer = NdjsonDatasetWriter(Path("output.ndjson"))
        >>> writer.write(normalized_dataset)
    """

    @abstractmethod
    def write(self, dataset: NormalizedDataset, output_path: Path) -> None:
        """Write a complete NormalizedDataset to storage.

        Args:
            dataset: The NormalizedDataset to write (samples + manifest)
            output_path: Path to output file

        Raises:
            IOError: If writing fails
        """
        pass
