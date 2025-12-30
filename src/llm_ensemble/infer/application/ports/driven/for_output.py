"""Port interface for writing LLM judgements.

Defines the abstract contract for writing judgements to various sinks
(JSON files, Parquet, databases, etc.). This allows the orchestrator
to work with any output format without coupling to a specific implementation.

Note: File-based adapters receive their output destination (run_dir) at
construction time via the factory, not through this interface. This keeps
the port interface clean and infrastructure-agnostic.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

from llm_ensemble.infer.domain.entities.llm_judgement import LLMJudgement
from llm_ensemble.infer.domain.entities.infer_run import InferRun
from llm_ensemble.infer.application.write_summary import WriteSummary


class ForOutput(ABC):
    """Abstract base class for writing LLM judgements with streaming support.

    Implementations can write to different sinks (JSON, Parquet, etc.)
    while providing a consistent interface to the domain service.

    Uses context manager pattern for proper resource lifecycle management.
    Each judgement is written immediately upon calling write_one(), enabling
    fault-tolerant streaming with partial progress preservation.

    The write summary is captured when close() is called and can be retrieved
    after the context manager exits via get_summary().

    InferRun is provided once during open() with config but no output yet.
    Output is finalized in close().

    Note: Output destination (e.g., run_dir for file-based adapters) is
    provided at construction time, not through this interface.
    """

    @property
    @abstractmethod
    def io_name(self) -> str:
        """Get I/O adapter name for this output port.

        Returns:
            I/O adapter name (e.g., 'json', 'parquet')
        """
        pass

    @abstractmethod
    def open(
        self,
        infer_run: InferRun,
    ) -> "ForOutput":
        """Initialize writer with InferRun aggregate root.

        InferRun contains:
        - Metadata: run_name, git_info, notes, run_type
        - Configuration: infer_run_config (model, adapters, execution context)
        - Output: infer_run_output (None at open time, set at close)

        For SQL writers, this creates the InferRun entity and prepares for streaming writes.

        Args:
            infer_run: InferRun aggregate root (config present, output=None)

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

        Adapter handles its own logging of what entities were written.

        Args:
            judgement: LLMJudgement object to write

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

    def __enter__(self) -> ForOutput:
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
