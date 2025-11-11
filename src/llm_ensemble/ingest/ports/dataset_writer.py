"""Port interface for writing judging samples.

Defines the abstract contract for writing judging samples to persistent storage.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from llm_ensemble.ingest.schemas import JudgingSample, WriteSummary
from llm_ensemble.ingest.schemas.ingest_run_info import IngestRunInfo


class DatasetWriter(ABC):
    """Abstract base class for writing judging samples.

    Writers are responsible for determining output file structure within the run directory.
    This allows different adapters to use their own naming conventions and formats.

    Writers return WriteSummary objects to provide transparency into write operations
    without handling their own logging (separation of concerns).

    Writers receive run_info separately from samples to maintain clean domain entities.
    This follows the separation of concerns where run context is passed to persistence
    adapters but not embedded in domain entities.

    Unlike INFER CLI which uses streaming writes (write_one in a loop), INGEST uses
    batch writes (write all samples at once), so no context manager pattern is needed.

    Example:
        >>> writer = FullyPopulatedNdjsonWriter()
        >>> summary = writer.write(samples, run_dir, run_info)
        >>> logger.info("write_complete", created=summary.total_created)
    """

    @abstractmethod
    def write(self, samples: List[JudgingSample], run_dir: Path, run_info: IngestRunInfo) -> WriteSummary:
        """Write judging samples to storage within the run directory.

        The adapter determines the specific output file(s) structure.
        Common patterns:
        - Separate files: run_dir / "judging_samples.ndjson" + run_dir / "ingest_run_info.json"
        - Database: Write to centralized database with foreign keys
        
        Filenames should be derived using get_entity_filename() for consistency
        with other CLIs (e.g., INFER uses llm_judgements.ndjson + infer_run_info.json).

        Args:
            samples: List of judging samples to write (pure domain entities without run_info)
            run_dir: Run directory where output should be written
            run_info: Immutable runtime context (written as separate manifest, not embedded in samples)

        Returns:
            WriteSummary tracking what was created vs. skipped

        Raises:
            IOError: If writing fails
        """
        pass
