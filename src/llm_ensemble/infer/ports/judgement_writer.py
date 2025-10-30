"""Port interface for writing LLM judgements.

Defines the abstract contract for writing judgements to various sinks
(NDJSON files, Parquet, databases, etc.). This allows the orchestrator
to work with any output format without coupling to a specific implementation.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path

from llm_ensemble.infer.schemas.llm_judgement import LLMJudgement


class JudgementWriter(ABC):
    """Abstract base class for writing LLM judgements with streaming support.

    Implementations can write to different sinks (NDJSON, Parquet, etc.)
    while providing a consistent interface to the domain service.

    Uses context manager pattern for proper resource lifecycle management.
    Each judgement is written immediately upon calling write_one(), enabling
    fault-tolerant streaming with partial progress preservation.

    Example:
        >>> writer = NdjsonJudgementWriter()
        >>> with writer.open(run_dir) as w:
        ...     for judgement in judgements:
        ...         w.write_one(judgement)  # Written immediately to disk
    """

    @abstractmethod
    def open(self, run_dir: Path) -> "JudgementWriter":
        """Initialize writer with run directory and prepare for streaming.

        Args:
            run_dir: Run directory where output should be written (writer determines file structure)

        Returns:
            Self, to enable context manager usage

        Raises:
            IOError: If writer cannot be initialized
        """
        pass

    @abstractmethod
    def write_one(self, judgement: LLMJudgement) -> None:
        """Write a single judgement to the output sink.

        Must be called within the context manager (after open()).
        Judgement is persisted immediately for fault tolerance.

        Args:
            judgement: LLMJudgement object to write

        Raises:
            IOError: If write operation fails
            RuntimeError: If called outside of context manager
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close writer and finalize output.

        Ensures all buffered data is flushed and resources are released.
        Called automatically by context manager __exit__.

        Raises:
            IOError: If close operation fails
        """
        pass

    def __enter__(self) -> "JudgementWriter":
        """Enter context manager.

        Returns:
            Self, for use in with statement
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager and ensure cleanup.

        Args:
            exc_type: Exception type if an error occurred
            exc_val: Exception value if an error occurred
            exc_tb: Exception traceback if an error occurred
        """
        self.close()
