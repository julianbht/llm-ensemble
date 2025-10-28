"""Port interface for writing judging samples.

Defines the abstract contract for writing judging samples to persistent storage.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from llm_ensemble.ingest.schemas import JudgingSample


class DatasetWriter(ABC):
    """Abstract base class for writing judging samples.

    Example:
        >>> writer = FullyPopulatedNdjsonWriter()
        >>> writer.write(samples, Path("output.ndjson"))
    """

    @abstractmethod
    def write(self, samples: List[JudgingSample], output_path: Path) -> None:
        """Write judging samples to storage.

        Args:
            samples: List of judging samples to write
            output_path: Path to output file

        Raises:
            IOError: If writing fails
        """
        pass
