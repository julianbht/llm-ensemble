"""Port interface for writing ingest run results.

Defines the abstract contract for persisting ingest run results to storage.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

from llm_ensemble.ingest.domain.entities.write_summary import WriteSummary
from llm_ensemble.ingest.domain.entities.ingest_run import IngestRun


class ForOutput(ABC):
    """Abstract base class for writing ingest run results.

    Writers are instantiated with io_name and run_dir, and receive the complete
    IngestRun aggregate at write time.

    Writers return WriteSummary objects to provide transparency into write operations
    without handling their own logging (separation of concerns).

    Writers receive an IngestRun aggregate root that contains:
    - ingest_run_config: The configuration used
    - normalized_dataset: The output dataset produced
    - Execution metadata: timing, git SHA, run name, notes

    This aggregate makes relationships explicit in the domain layer, avoiding
    the need to pass disconnected objects and manually inject foreign key IDs.

    Unlike INFER CLI which uses streaming writes (write_one in a loop), INGEST uses
    batch writes (write all samples at once), so no context manager pattern is needed.
    """

    @abstractmethod
    def write(self, ingest_run: IngestRun) -> WriteSummary:
        """Write ingest run results to storage.

        The adapter determines output location and structure:
        - File-based writers: Use self.run_dir for output directory
        - Database writers: Write to centralized database

        The ingest_run aggregate contains everything needed:
        - ingest_run.normalized_dataset: Dataset with samples
        - ingest_run.ingest_run_config: Config used to create dataset
        - ingest_run.start_time, end_time: Timing information
        - ingest_run.git_sha, git_branch, etc.: Git metadata

        Args:
            ingest_run: Complete IngestRun aggregate root

        Returns:
            WriteSummary tracking what was created vs. skipped

        Raises:
            IOError: If writing fails
        """
        pass
