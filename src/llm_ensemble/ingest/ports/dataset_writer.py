"""Port interface for writing judging samples.

Defines the abstract contract for writing judging samples to persistent storage.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from llm_ensemble.ingest.schemas import JudgingSample, WriteSummary


class DatasetWriter(ABC):
    """Abstract base class for writing judging samples.

    Writers are responsible for determining output file structure within the run directory.
    This allows different adapters to use their own naming conventions and formats.

    Writers return WriteSummary objects to provide transparency into write operations
    without handling their own logging (separation of concerns).

    Example:
        >>> writer = FullyPopulatedNdjsonWriter()
        >>> summary = writer.write(samples, Path("artifacts/runs/ingest/test/20250128_123456_dataset"))
        >>> logger.info("write_complete", created=summary.total_created, skipped=summary.total_skipped)
    """

    @abstractmethod
    def write(self, samples: List[JudgingSample], run_dir: Path) -> WriteSummary:
        """Write judging samples to storage within the run directory.

        The adapter determines the specific output file(s) structure.
        Common patterns:
        - Single file: run_dir / "normalized_dataset.ndjson"
        - Multiple files: run_dir / "samples.ndjson", run_dir / "metadata.json"

        Args:
            samples: List of judging samples to write
            run_dir: Run directory where output should be written

        Returns:
            WriteSummary tracking what was created vs. skipped

        Raises:
            IOError: If writing fails
        """
        pass
