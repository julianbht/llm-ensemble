"""Port interface for writing judging samples.

Defines the abstract contract for writing judging samples to persistent storage.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

from llm_ensemble.ingest.domain.entities.write_summary import WriteSummary
from llm_ensemble.ingest.domain.entities.ingest_run_info import IngestRunInfo


class ForOutput(ABC):
    """Abstract base class for writing judging samples.

    Writers are instantiated with io_name and run_dir, and receive the complete
    IngestRunInfo aggregate at write time.

    Writers return WriteSummary objects to provide transparency into write operations
    without handling their own logging (separation of concerns).

    Writers receive an IngestRunInfo aggregate root that embeds:
    - normalized_dataset: The output dataset (which embeds run_config)
    - Run metadata: git SHA, timestamps, run name, notes

    This aggregate makes relationships explicit in the domain layer, avoiding
    the need to pass disconnected objects and manually inject foreign key IDs.

    Unlike INFER CLI which uses streaming writes (write_one in a loop), INGEST uses
    batch writes (write all samples at once), so no context manager pattern is needed.
    """

    @abstractmethod
    def write(self, run_info: IngestRunInfo) -> WriteSummary:
        """Write judging samples to storage.

        The adapter determines output location and structure:
        - File-based writers: Use self.run_dir for output directory
        - Database writers: Write to centralized database

        The run_info aggregate contains everything needed:
        - run_info.normalized_dataset: Dataset with samples
        - run_info.normalized_dataset.run_config: Config used to create dataset
        - run_info.git_info, run_info.run_name, etc.: Runtime metadata

        Args:
            run_info: Complete IngestRunInfo aggregate root

        Returns:
            WriteSummary tracking what was created vs. skipped

        Raises:
            IOError: If writing fails
        """
        pass
