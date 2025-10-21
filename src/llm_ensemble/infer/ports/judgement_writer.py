"""Port interface for writing model judgements.

Defines the abstract contract for writing judgements to various sinks
(NDJSON files, Parquet, databases, etc.). This allows the orchestrator
to work with any output format without coupling to a specific implementation.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

from llm_ensemble.infer.schemas import ModelJudgement


class JudgementWriter(ABC):
    """Abstract base class for writing model judgements.

    Implementations can write to different sinks (NDJSON, Parquet, etc.)
    while providing a consistent interface to the orchestrator.

    The writer should be used as a context manager or with explicit close():

    Example:
        >>> writer = NdjsonJudgementWriter(output_path)
        >>> for judgement in judgements:
        ...     writer.write(judgement)
        >>> writer.close()
    """

    @abstractmethod
    def write(self, judgement: ModelJudgement) -> None:
        """Write a single judgement to the output sink.

        Args:
            judgement: ModelJudgement object to write

        Raises:
            IOError: If write operation fails
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Finalize and close the output sink.

        This method should flush any buffered data and release resources.
        It's safe to call multiple times (idempotent).
        """
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure resources are closed."""
        self.close()
        return False
