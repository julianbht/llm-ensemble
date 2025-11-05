"""Port interface for writing LLM judgements.

Defines the abstract contract for writing judgements to various sinks
(NDJSON files, Parquet, databases, etc.). This allows the orchestrator
to work with any output format without coupling to a specific implementation.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from llm_ensemble.infer.schemas.llm_judgement import LLMJudgement
from llm_ensemble.infer.schemas.write_summary import WriteSummary
from llm_ensemble.libs.schemas.write_result import WriteResult


class JudgementWriter(ABC):
    """Abstract base class for writing LLM judgements with streaming support.

    Implementations can write to different sinks (NDJSON, Parquet, etc.)
    while providing a consistent interface to the domain service.

    Uses context manager pattern for proper resource lifecycle management.
    Each judgement is written immediately upon calling write_one(), enabling
    fault-tolerant streaming with partial progress preservation.

    The write summary is captured when close() is called and can be retrieved
    after the context manager exits via get_summary().

    Example:
        >>> writer = NdjsonJudgementWriter()
        >>> with writer.open(run_dir) as w:
        ...     for judgement in judgements:
        ...         w.write_one(judgement)  # Written immediately to disk
        >>> summary = writer.get_summary()  # Get write summary
    """

    def __init__(self):
        """Initialize writer."""
        self._write_summary: Optional[WriteSummary] = None

    @abstractmethod
    def open(self, run_dir: Path) -> JudgementWriter:
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
    def write_one(self, judgement: LLMJudgement) -> WriteResult:
        """Write a single judgement to the output sink.

        Must be called within the context manager (after open()).
        Judgement is persisted immediately for fault tolerance.

        Args:
            judgement: LLMJudgement object to write

        Returns:
            WriteResult for this specific write operation (contains item ID and type)

        Raises:
            IOError: If write operation fails
            RuntimeError: If called outside of context manager
        """
        pass

    @abstractmethod
    def close(self) -> WriteSummary:
        """Close writer and finalize output.

        Ensures all buffered data is flushed and resources are released.
        Called automatically by context manager __exit__.

        Returns:
            WriteSummary tracking write operations performed during streaming

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

        Calls close() and stores the returned WriteSummary for later retrieval.

        Args:
            exc_type: Exception type if an error occurred
            exc_val: Exception value if an error occurred
            exc_tb: Exception traceback if an error occurred
        """
        self._write_summary = self.close()

    def get_summary(self) -> WriteSummary:
        """Get the write summary after the writer has been closed.

        Must be called after the context manager exits.

        Returns:
            WriteSummary tracking write operations

        Raises:
            RuntimeError: If called before writer is closed
        """
        if self._write_summary is None:
            raise RuntimeError("Writer has not been closed yet - must call after context manager exits")
        return self._write_summary
