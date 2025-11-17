"""Port interface for reading judging examples.

Defines the abstract contract for reading examples from various sources
(JSON files, Parquet, databases, etc.). This allows the orchestrator
to work with any input format without coupling to a specific implementation.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

from llm_ensemble.ingest.schemas import JudgingSample


class ExampleReader(ABC):
    """Abstract base class for reading judging examples.

    Implementations can read from different sources (JSON, Parquet, etc.)
    while providing a consistent interface to the orchestrator.
    
    Readers accept run_name strings and internally resolve to file paths
    using PathManager, enabling clean separation of concerns.
    """

    @abstractmethod
    def read(
        self,
        run_name: str,
        limit: Optional[int] = None,
    ) -> list[JudgingSample]:
        """Read examples from source.

        Args:
            run_name: Run identifier (e.g., "my_ingest_run" or "20250128_143022")
                     Readers use PathManager to resolve to appropriate file paths
            limit: Optional maximum number of examples to read

        Returns:
            List of JudgingSample objects

        Raises:
            FileNotFoundError: If run directory or expected files don't exist
            ValueError: If run name is invalid or data is malformed
        """
        pass
