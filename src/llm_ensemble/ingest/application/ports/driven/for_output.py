"""Port interface for writing judging samples.

Defines the abstract contract for writing judging samples to persistent storage.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

from llm_ensemble.ingest.domain.entities.write_summary import WriteSummary
from llm_ensemble.ingest.domain.entities.normalized_dataset import NormalizedDataset
from llm_ensemble.ingest.domain.entities.ingest_run_info import IngestRunInfo
from llm_ensemble.ingest.domain.entities.ingest_run_config import IngestRunConfig


class ForOutput(ABC):
    """Abstract base class for writing judging samples.

    Writers are responsible for determining output file structure within the run directory.
    This allows different adapters to use their own naming conventions and formats.

    Writers return WriteSummary objects to provide transparency into write operations
    without handling their own logging (separation of concerns).

    Writers receive run_info and run_config separately from normalized dataset to maintain
    clean domain entities. This follows separation of concerns:
    - run_info: Runtime metadata (git SHA, timestamps, notes)
    - run_config: Configuration used for this run (I/O config, input path, limit)

    Unlike INFER CLI which uses streaming writes (write_one in a loop), INGEST uses
    batch writes (write all samples at once), so no context manager pattern is needed.
    """

    @abstractmethod
    def write(
        self,
        normalized_dataset: NormalizedDataset,
        run_info: IngestRunInfo,
        run_config: IngestRunConfig,
    ) -> WriteSummary:
        """Write judging samples to storage.

        The adapter determines output location and structure:
        - File-based writers: Use run_info.run_dir for output directory
        - Database writers: Write to centralized database

        Filenames should be derived using get_entity_filename() for consistency
        with other CLIs (e.g., INFER uses llm_judgements.json + infer_run_info.json).

        Args:
            normalized_dataset: Complete normalized dataset with samples and metadata
            run_info: Immutable runtime metadata (git SHA, timestamps, notes)
            run_config: Immutable run configuration (I/O config, input path, limit)

        Returns:
            WriteSummary tracking what was created vs. skipped

        Raises:
            IOError: If writing fails
        """
        pass
