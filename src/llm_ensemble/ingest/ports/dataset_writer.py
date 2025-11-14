"""Port interface for writing judging samples.

Defines the abstract contract for writing judging samples to persistent storage.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from llm_ensemble.ingest.schemas import WriteSummary, NormalizedDataset
from llm_ensemble.ingest.schemas.ingest_run_info import IngestRunInfo


class DatasetWriter(ABC):
    """Abstract base class for writing judging samples.

    Writers are responsible for determining output file structure within the run directory.
    This allows different adapters to use their own naming conventions and formats.

    Writers return WriteSummary objects to provide transparency into write operations
    without handling their own logging (separation of concerns).

    Writers receive run_info separately from normalized dataset to maintain clean domain entities.
    This follows the separation of concerns where run context is passed to persistence
    adapters but not embedded in domain entities.

    Unlike INFER CLI which uses streaming writes (write_one in a loop), INGEST uses
    batch writes (write all samples at once), so no context manager pattern is needed.

    Example:
        >>> writer = FullyPopulatedNdjsonWriter()
        >>> summary = writer.write(normalized_dataset, run_info)
        >>> logger.info("write_complete", created=summary.total_created)
    """

    @abstractmethod
    def write(
        self,
        normalized_dataset: NormalizedDataset,
        run_info: IngestRunInfo,
    ) -> WriteSummary:
        """Write judging samples to storage.

        The adapter determines output location and structure:
        - File-based writers: Use run_info.run_dir for output directory
        - Database writers: Write to centralized database

        Filenames should be derived using get_entity_filename() for consistency
        with other CLIs (e.g., INFER uses llm_judgements.json + infer_run_info.json).

        Args:
            normalized_dataset: Complete normalized dataset with samples and metadata
            run_info: Immutable runtime context (contains run_dir property for path derivation)

        Returns:
            WriteSummary tracking what was created vs. skipped

        Raises:
            IOError: If writing fails
        """
        pass
