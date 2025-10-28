"""Port interface for writing judging samples with manifest metadata.

Defines the abstract contract for writing judging samples to persistent storage.
Each sample carries a reference to its manifest (Many-to-One relationship).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from llm_ensemble.ingest.schemas import JudgingSample, IngestManifest


class DatasetWriter(ABC):
    """Abstract base class for writing judging samples.

    Writes judging samples to persistent storage. Each sample contains a reference
    to the manifest, establishing a Many-to-One relationship between samples and manifest.

    Example:
        >>> writer = NdjsonDatasetWriter()
        >>> writer.write(samples, manifest, Path("output.ndjson"))
    """

    @abstractmethod
    def write(
        self,
        samples: List[JudgingSample],
        manifest: IngestManifest,
        output_path: Path
    ) -> None:
        """Write judging samples with manifest metadata to storage.

        Args:
            samples: List of judging samples (each contains manifest reference)
            manifest: The ingest manifest for metadata and quick reference
            output_path: Path to output file

        Raises:
            IOError: If writing fails
        """
        pass
