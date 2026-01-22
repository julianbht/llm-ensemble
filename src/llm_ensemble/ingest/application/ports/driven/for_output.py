"""Port interface for writing ingest run results.

Defines the abstract contract for persisting ingest run results to storage.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

from llm_ensemble.ingest.domain.entities.write_summary import WriteSummary
from llm_ensemble.ingest.domain.entities.ingest_run import IngestRun


class ForOutput(ABC):
    """
    Abstract base class for writing ingest run results.
    """

    @abstractmethod
    def write(self, ingest_run: IngestRun) -> WriteSummary:
        """Write ingest run results to storage.

        Args:
            ingest_run: Complete IngestRun aggregate root

        Returns:
            WriteSummary tracking what was created vs. skipped

        Raises:
            IOError: If writing fails
        """
        pass
